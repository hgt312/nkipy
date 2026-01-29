// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/EmptyTensor.h>
#include <ATen/InferSize.h>
#include <ATen/native/Resize.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <nrt/nrt.h>

#include <algorithm>

#include "nkipy_device.h"
#include "nkipy_tensor.h"
#include "nkipy_tensor_allocator.h"

namespace nkipy_tensor {

NKIPyStorageImpl::NKIPyStorageImpl(use_byte_size_t use_byte_size,
                                   size_t size_bytes, at::DataPtr data_ptr,
                                   at::Allocator *allocator, bool resizable)
    : c10::StorageImpl(use_byte_size, size_bytes, std::move(data_ptr),
                       allocator, resizable) {
  // The nkipy tensor is stored in the DataPtr context
  // We'll retrieve it when needed through the public API
}

void NKIPyStorageImpl::release_resources() {
  // The DataPtr deleter will handle freeing the nkipy tensor
  // We just need to call the base class implementation
  StorageImpl::release_resources();
}

NKIPyTensorImpl::NKIPyTensorImpl(c10::Storage &&storage,
                                 const caffe2::TypeMeta &data_type)
    : c10::TensorImpl(std::move(storage),
                      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                      data_type) {
  // Set nkipy-specific dispatch keys
  SetNKIPyDispatchKeys();
}

void NKIPyTensorImpl::SetNKIPyDispatchKeys() {
  // Set the dispatch keys for Neuron device
  key_set_ = key_set_.add(c10::DispatchKey::PrivateUse1);
  key_set_ = key_set_.add(c10::DispatchKey::AutogradPrivateUse1);
}

void NKIPyTensorImpl::shallow_copy_from(
    const c10::intrusive_ptr<TensorImpl> &impl) {
  TensorImpl::shallow_copy_from(impl);
  SetNKIPyDispatchKeys();
}

c10::intrusive_ptr<c10::TensorImpl> NKIPyTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion &version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl =
      c10::make_intrusive<NKIPyTensorImpl>(c10::Storage(storage()), data_type_);

  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);

  impl->refresh_numel();
  impl->refresh_contiguous();
  return impl;
}

c10::intrusive_ptr<c10::TensorImpl> NKIPyTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion &&version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl =
      c10::make_intrusive<NKIPyTensorImpl>(c10::Storage(storage()), data_type_);

  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/std::move(version_counter),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);

  impl->refresh_numel();
  impl->refresh_contiguous();
  return impl;
}

// Helper to create Neuron storage
c10::Storage make_nkipy_storage(size_t size_bytes, int device_index) {
  // Get the allocator
  c10::Allocator *allocator = nkipy_tensor_allocator::get();

  // Set the current device for the allocator
  nkipy_device::set_device(device_index);

  // Allocate through the allocator
  auto data_ptr = allocator->allocate(size_bytes);

  // Create storage implementation with allocator
  auto storage_impl = c10::make_intrusive<NKIPyStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes, std::move(data_ptr),
      allocator, // Pass the allocator (changed from nullptr)
      true);     // resizable = true to support storage resize

  return c10::Storage(std::move(storage_impl));
}

at::Tensor empty_memory_format_nkipy(
    c10::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt, c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  // Check memory format
  auto memory_format =
      memory_format_opt.value_or(c10::MemoryFormat::Contiguous);
  TORCH_CHECK(
      memory_format == c10::MemoryFormat::Contiguous,
      "Neuron backend only supports contiguous memory format, but got: ",
      memory_format);
  return empty_strided_nkipy(size, c10::TensorType::contiguousStridesOf(size),
                             dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

at::Tensor empty_strided_nkipy(at::IntArrayRef size, at::IntArrayRef stride,
                               c10::optional<c10::ScalarType> dtype_opt,
                               c10::optional<c10::Layout> layout_opt,
                               c10::optional<c10::Device> device_opt,
                               c10::optional<bool> pin_memory_opt) {
  // Validate device
  auto device = c10::device_or_default(device_opt);
  TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1,
              "Expected nkipy device but got: ", device);

  // Get device index
  int device_index = device.has_index() ? device.index() : 0;

  // Get dtype
  auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(dtype_opt));

  // Calculate total size
  int64_t nelements = c10::multiply_integers(size);
  int64_t size_bytes = nelements * dtype.itemsize();

  // Handle zero-size tensors
  if (size_bytes == 0) {
    // For zero-size tensors, we still need to create a valid tensor
    // but with minimal allocation
    size_bytes = dtype.itemsize();
  }

  // Create storage
  auto storage = make_nkipy_storage(size_bytes, device_index);

  // Create tensor
  auto tensor =
      at::detail::make_tensor<NKIPyTensorImpl>(std::move(storage), dtype);

  tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return tensor;
}

const at::Tensor &resize_nkipy(const at::Tensor &self, c10::IntArrayRef size,
                               std::optional<c10::MemoryFormat> memory_format) {
  // Validate device
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "resize_: expected nkipy tensor, but got: ", self.device());

  // Check memory format
  auto format = memory_format.value_or(c10::MemoryFormat::Contiguous);
  TORCH_CHECK(format == c10::MemoryFormat::Contiguous ||
                  format == c10::MemoryFormat::Preserve,
              "Neuron backend only supports contiguous or preserve memory "
              "format, but got: ",
              format);

  // If preserve, use contiguous
  if (format == c10::MemoryFormat::Preserve) {
    format = c10::MemoryFormat::Contiguous;
  }

  // Calculate new total elements and size
  int64_t new_nelements = c10::multiply_integers(size);
  auto dtype = self.scalar_type();
  size_t itemsize = c10::elementSize(dtype);
  size_t new_size_bytes = new_nelements * itemsize;

  // Get current storage size
  size_t current_size = self.storage().nbytes();

  // Check if we need to reallocate
  if (new_size_bytes > current_size) {
    // Need larger storage
    int device_index = self.device().index();

    // Set device context
    nkipy_device::set_device(device_index);

    // Allocate new storage
    auto new_storage = make_nkipy_storage(new_size_bytes, device_index);

    // Copy existing data if any
    if (self.numel() > 0 && current_size > 0) {
      void *old_data = self.data_ptr();
      void *new_data = new_storage.data_ptr().get();

      // Copy min of old and new size
      size_t copy_size = std::min(current_size, new_size_bytes);
      bool success =
          nkipy_tensor_allocator::copyTensorData(new_data, old_data, copy_size);
      TORCH_CHECK(success, "Failed to copy data during resize");
    }

    // Replace storage
    self.unsafeGetTensorImpl()->set_storage_and_dtype(
        new_storage, c10::scalarTypeToTypeMeta(dtype));
  }

  // Update sizes and strides
  if (new_nelements > 0) {
    self.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  } else {
    // For zero-size tensors, manually compute strides
    std::vector<int64_t> strides(size.size());
    int64_t stride = 1;
    for (int64_t i = size.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= size[i];
    }
    self.unsafeGetTensorImpl()->set_sizes_and_strides(size, strides);
  }

  return self;
}

nrt_tensor_t *get_nrt_tensor(const at::Tensor &tensor) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              "Expected Neuron tensor, got ", tensor.device());

  // Get the storage implementation
  auto storage_impl = tensor.storage().unsafeGetStorageImpl();
  auto nkipy_storage = static_cast<NKIPyStorageImpl *>(storage_impl);

  return nkipy_storage->nrt_tensor();
}

void copy_cpu_to_nkipy(const at::Tensor &src, at::Tensor &dst) {
  TORCH_CHECK(src.is_cpu(), "Source tensor must be CPU tensor");
  TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
              "Destination must be Neuron tensor");

  TORCH_CHECK(
      src.is_contiguous(),
      "Source CPU tensor must be contiguous. "
      "Please call .contiguous() on CPU tensor before copying to Neuron");
  TORCH_CHECK(dst.is_contiguous(),
              "Destination Neuron tensor must be contiguous");

  nrt_tensor_t *nrt_dst = get_nrt_tensor(dst);
  TORCH_CHECK(nrt_dst != nullptr, "Failed to get Neuron tensor handle");

  // Calculate offset based on tensor's storage offset
  size_t byte_offset = dst.storage_offset() * dst.element_size();

  // Skip copy if there are no bytes to copy (empty tensor)
  if (src.nbytes() == 0) {
    return;
  }

  NRT_STATUS status =
      nrt_tensor_write(nrt_dst, src.data_ptr(), byte_offset, src.nbytes());

  TORCH_CHECK(status == NRT_SUCCESS,
              "Failed to copy data to Neuron device. Status: ", status);
}

void copy_nkipy_to_cpu(const at::Tensor &src, at::Tensor &dst) {
  TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
              "Source must be Neuron tensor");
  TORCH_CHECK(dst.is_cpu(), "Destination tensor must be CPU tensor");

  // If source is not contiguous but we need contiguous data, throw an error
  at::Tensor src_to_copy = src;
  // FIXME
  if (!src.is_contiguous() && dst.is_contiguous()) {
    TORCH_CHECK(false,
                "Source Neuron tensor must be contiguous for copy to CPU");
  }

  TORCH_CHECK(
      dst.is_contiguous(),
      "Destination CPU tensor must be contiguous for direct copy from Neuron");

  nrt_tensor_t *nrt_src = get_nrt_tensor(src_to_copy);
  TORCH_CHECK(nrt_src != nullptr, "Failed to get Neuron tensor handle");

  // Calculate offset based on tensor's storage offset
  size_t byte_offset =
      src_to_copy.storage_offset() * src_to_copy.element_size();

  // Skip copy if there are no bytes to copy (empty tensor)
  if (dst.nbytes() == 0) {
    return;
  }

  NRT_STATUS status =
      nrt_tensor_read(nrt_src, dst.data_ptr(), byte_offset, dst.nbytes());

  TORCH_CHECK(status == NRT_SUCCESS,
              "Failed to copy data from Neuron device. Status: ", status);
}

void copy_nkipy_to_nkipy(const at::Tensor &src, at::Tensor &dst) {
  TORCH_CHECK(src.device().type() == c10::DeviceType::PrivateUse1,
              "Source must be Neuron tensor");
  TORCH_CHECK(dst.device().type() == c10::DeviceType::PrivateUse1,
              "Destination must be Neuron tensor");

  // Check if on same device (required by nrt_tensor_copy for D2D)
  TORCH_CHECK(src.device().index() == dst.device().index(),
              "Device-to-device copy requires tensors on same Neuron core");

  // If source is not contiguous but we need contiguous data, make it contiguous
  // first
  at::Tensor src_to_copy = src;
  // FIXME
  if (!src.is_contiguous() && dst.is_contiguous()) {
    TORCH_CHECK(false,
                "Source Neuron tensor must be contiguous for copy to nkipy");
  }

  TORCH_CHECK(dst.is_contiguous(),
              "Destination Neuron tensor must be contiguous");

  nrt_tensor_t *nrt_src = get_nrt_tensor(src_to_copy);
  nrt_tensor_t *nrt_dst = get_nrt_tensor(dst);
  TORCH_CHECK(nrt_src != nullptr, "Failed to get source Neuron tensor handle");
  TORCH_CHECK(nrt_dst != nullptr,
              "Failed to get destination Neuron tensor handle");

  // Calculate offsets based on tensor's storage offset
  size_t src_byte_offset =
      src_to_copy.storage_offset() * src_to_copy.element_size();
  size_t dst_byte_offset = dst.storage_offset() * dst.element_size();

  // Ensure tensors have same size
  TORCH_CHECK(src_to_copy.nbytes() == dst.nbytes(),
              "Source and destination must have same number of bytes");

  // Skip copy if there are no bytes to copy (empty tensor)
  if (src_to_copy.nbytes() == 0) {
    return;
  }

  NRT_STATUS status = nrt_tensor_copy(nrt_src, src_byte_offset, // src + offset
                                      nrt_dst, dst_byte_offset, // dst + offset
                                      src_to_copy.nbytes());

  TORCH_CHECK(status == NRT_SUCCESS,
              "Failed to copy between Neuron tensors. Status: ", status);
}

at::Tensor &copy_nkipy(at::Tensor &self, const at::Tensor &src,
                       bool non_blocking) {
  if (self.is_same(src)) {
    return self;
  }

  // Check sizes match
  TORCH_CHECK(self.sizes() == src.sizes(),
              "Source and destination sizes must match. Got src: ", src.sizes(),
              " and dst: ", self.sizes());

  bool same_type = self.dtype() == src.dtype();

  // CPU to Neuron
  if (src.is_cpu() && self.device().type() == c10::DeviceType::PrivateUse1) {
    if (!same_type) {
      // Convert dtype on CPU side first
      at::Tensor src_converted = src.to(self.dtype());
      copy_cpu_to_nkipy(src_converted, self);
    } else {
      copy_cpu_to_nkipy(src, self);
    }
    return self;
  }

  // Neuron to CPU
  if (src.device().type() == c10::DeviceType::PrivateUse1 && self.is_cpu()) {
    if (!same_type) {
      // Create temporary buffer with source dtype, then convert on CPU
      at::Tensor temp_cpu =
          at::empty(src.sizes(), self.options().dtype(src.dtype()));
      copy_nkipy_to_cpu(src, temp_cpu);
      self.copy_(temp_cpu); // This will use PyTorch's native CPU copy with
                            // dtype conversion
    } else {
      copy_nkipy_to_cpu(src, self);
    }
    return self;
  }

  // Neuron to Neuron
  if (src.device().type() == c10::DeviceType::PrivateUse1 &&
      self.device().type() == c10::DeviceType::PrivateUse1) {
    if (!same_type) {
      // For now, go through CPU for dtype conversion
      // TODO: Implement direct device dtype conversion when available
      at::Tensor temp_cpu = src.to(at::kCPU, src.dtype());
      temp_cpu = temp_cpu.to(self.dtype());
      copy_cpu_to_nkipy(temp_cpu, self);
    } else {
      copy_nkipy_to_nkipy(src, self);
    }
    return self;
  }

  TORCH_CHECK(false, "Unsupported device combination for copy_");
}

at::Tensor _to_copy_nkipy(const at::Tensor &self,
                          c10::optional<at::ScalarType> dtype,
                          c10::optional<c10::Layout> layout,
                          c10::optional<c10::Device> device,
                          c10::optional<bool> pin_memory, bool non_blocking,
                          c10::optional<c10::MemoryFormat> memory_format) {
  // Validate memory format for Neuron tensors
  bool is_to_nkipy =
      device.has_value() && device->type() == c10::DeviceType::PrivateUse1;
  bool is_from_nkipy = self.device().type() == c10::DeviceType::PrivateUse1;

  if ((is_to_nkipy || is_from_nkipy) && memory_format.has_value() &&
      *memory_format != c10::MemoryFormat::Contiguous &&
      *memory_format != c10::MemoryFormat::Preserve) {
    TORCH_CHECK(false, "Neuron tensors only support contiguous memory format");
  }

  // Build target options
  auto options = self.options();
  if (dtype.has_value())
    options = options.dtype(*dtype);
  if (layout.has_value())
    options = options.layout(*layout);
  if (device.has_value())
    options = options.device(*device);
  if (pin_memory.has_value())
    options = options.pinned_memory(*pin_memory);

  // Check for no-op
  bool needs_copy = false;
  if (dtype.has_value() && *dtype != self.dtype())
    needs_copy = true;
  if (device.has_value() && *device != self.device())
    needs_copy = true;
  if (layout.has_value() && *layout != self.layout())
    needs_copy = true;

  if (!needs_copy && (!memory_format.has_value() ||
                      *memory_format == c10::MemoryFormat::Preserve)) {
    return self;
  }

  // For CPU->Neuron transfers with non-contiguous source, we need special
  // handling
  if (is_to_nkipy && !self.is_contiguous()) {
    // First create a contiguous CPU copy, then copy to Neuron
    at::Tensor cpu_contig = self.contiguous();
    at::Tensor result = at::empty(cpu_contig.sizes(), options);
    result.copy_(cpu_contig, non_blocking);
    return result;
  }

  // Create result tensor
  at::Tensor result = at::empty(self.sizes(), options);

  // Copy data
  result.copy_(self, non_blocking);

  return result;
}

// Helper function to create a view tensor with new sizes and strides
static at::Tensor alias_with_sizes_and_strides(const at::Tensor &self,
                                               at::IntArrayRef sizes,
                                               at::IntArrayRef strides) {
  // Create a new tensor that shares storage with the original
  at::Tensor result = at::detail::make_tensor<at::TensorImpl>(
      c10::TensorImpl::VIEW,        // Mark as view
      c10::Storage(self.storage()), // Share storage
      self.key_set(), self.dtype());

  // Set the metadata
  auto *impl = result.unsafeGetTensorImpl();
  impl->set_storage_offset(self.storage_offset());
  impl->set_sizes_and_strides(sizes, strides);

  // Propagate names if any
  at::namedinference::propagate_names(result, self);

  return result;
}

// View operation - reshapes tensor without copying data
at::Tensor view_nkipy(const at::Tensor &self, at::IntArrayRef size) {
  // Infer size (handle -1 in size specification)
  auto inferred_size = at::infer_size(size, self.numel());

  // Compute strides for the new size
  auto stride =
      at::detail::computeStride(self.sizes(), self.strides(), inferred_size);

  TORCH_CHECK(stride.has_value(),
              "view size is not compatible with input tensor's size and stride "
              "(at least one dimension spans across two contiguous subspaces). "
              "Use .reshape(...) instead.");

  // Create the view
  return alias_with_sizes_and_strides(self, inferred_size, *stride);
}

// as_strided - the fundamental view operation
at::Tensor as_strided_nkipy(const at::Tensor &self, at::IntArrayRef size,
                            at::IntArrayRef stride,
                            c10::optional<int64_t> storage_offset_) {
  auto storage_offset = storage_offset_.value_or(self.storage_offset());

  // Validate the new view doesn't go out of bounds
  at::native::checkInBoundsForStorage(size, stride, storage_offset,
                                      self.dtype(), self.storage());

  // Create the strided view
  at::Tensor result = at::detail::make_tensor<at::TensorImpl>(
      c10::TensorImpl::VIEW, c10::Storage(self.storage()), self.key_set(),
      self.dtype());

  at::native::setStrided(result, size, stride, storage_offset);
  at::namedinference::propagate_names(result, self);

  return result;
}

// _reshape_alias - used internally by PyTorch for view operations
at::Tensor _reshape_alias_nkipy(const at::Tensor &self, at::IntArrayRef sizes,
                                at::IntArrayRef strides) {
  return view_nkipy(self, sizes);
}

at::Tensor _copy_from_and_resize_nkipy(const at::Tensor &self,
                                       const at::Tensor &dst) {
  // Validate that both tensors are defined
  TORCH_CHECK(dst.defined(), "dst is undefined");
  TORCH_CHECK(self.defined(), "self is undefined");

  // Check that we're copying from CPU to Neuron
  TORCH_CHECK(self.is_cpu() &&
                  dst.device().type() == c10::DeviceType::PrivateUse1,
              "_copy_from_and_resize now only support copy from cpu tensor to "
              "nkipy tensor, but "
              "got src tensor device is ",
              self.device(), " and dst device is ", dst.device());

  // If dst is empty (numel() == 0), resize it to match self
  if (dst.numel() == 0) {
    // Cast away const to resize - this is safe as we're modifying dst which is
    // the output
    const_cast<at::Tensor &>(dst).resize_as_(self);
  }

  // After potential resize, check that sizes match
  TORCH_CHECK(self.sizes() == dst.sizes(),
              "_copy_from_and_resize now only support copy with same size, or "
              "dst.numel() == 0!");

  // Perform the copy using the existing copy_ operation
  const_cast<at::Tensor &>(dst).copy_(self);

  return dst;
}

at::Tensor &fill_scalar_nkipy(at::Tensor &self, const at::Scalar &value) {
  // Validate device
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "fill_: expected nkipy tensor, but got: ", self.device());

  // Create a CPU tensor with the same shape and fill it
  at::Tensor cpu_tensor = at::empty_like(self, self.options().device(at::kCPU));
  cpu_tensor.fill_(value);

  // Copy the filled CPU tensor back to the Neuron tensor
  self.copy_(cpu_tensor);

  return self;
}

} // namespace nkipy_tensor
