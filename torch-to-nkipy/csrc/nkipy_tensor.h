// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <nrt/nrt.h>

namespace nkipy_tensor {

struct NKIPyStorageImpl : public c10::StorageImpl {
  NKIPyStorageImpl(use_byte_size_t use_byte_size, size_t size_bytes,
                   at::DataPtr data_ptr, at::Allocator *allocator,
                   bool resizable);

  ~NKIPyStorageImpl() override = default;

  void release_resources() override;

  // Initially we don't support resizing
  // We'll add this functionality later when we have an allocator

  // Getters
  nrt_tensor_t *nrt_tensor() const {
    // Get tensor from DataPtr context
    auto ctx = data_ptr().get_context();
    return ctx ? static_cast<nrt_tensor_t *>(ctx) : nullptr;
  }
};

class NKIPyTensorImpl : public c10::TensorImpl {
public:
  explicit NKIPyTensorImpl(c10::Storage &&storage,
                           const caffe2::TypeMeta &data_type);

  // Shallow copy operations
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl> &impl) override;
  c10::intrusive_ptr<c10::TensorImpl>
  shallow_copy_and_detach(const c10::VariableVersion &version_counter,
                          bool allow_tensor_metadata_change) const override;
  c10::intrusive_ptr<c10::TensorImpl>
  shallow_copy_and_detach(c10::VariableVersion &&version_counter,
                          bool allow_tensor_metadata_change) const override;

private:
  // Helper to set device-specific dispatch keys
  void SetNKIPyDispatchKeys();
};

//  Empty operations
at::Tensor empty_memory_format_nkipy(
    c10::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt, c10::optional<bool> pin_memory_opt,
    c10::optional<c10::MemoryFormat> memory_format_opt);

at::Tensor empty_strided_nkipy(at::IntArrayRef size, at::IntArrayRef stride,
                               c10::optional<c10::ScalarType> dtype_opt,
                               c10::optional<c10::Layout> layout_opt,
                               c10::optional<c10::Device> device_opt,
                               c10::optional<bool> pin_memory_opt);

// Resize operations
const at::Tensor &resize_nkipy(const at::Tensor &self, c10::IntArrayRef size,
                               std::optional<c10::MemoryFormat> memory_format);

at::Tensor &copy_nkipy(at::Tensor &self, const at::Tensor &src,
                       bool non_blocking = false);

at::Tensor _to_copy_nkipy(const at::Tensor &self,
                          c10::optional<at::ScalarType> dtype,
                          c10::optional<c10::Layout> layout,
                          c10::optional<c10::Device> device,
                          c10::optional<bool> pin_memory, bool non_blocking,
                          c10::optional<c10::MemoryFormat> memory_format);

// View operations
at::Tensor view_nkipy(const at::Tensor &self, at::IntArrayRef size);

at::Tensor as_strided_nkipy(const at::Tensor &self, at::IntArrayRef size,
                            at::IntArrayRef stride,
                            c10::optional<int64_t> storage_offset);

at::Tensor _reshape_alias_nkipy(const at::Tensor &self, at::IntArrayRef sizes,
                                at::IntArrayRef strides);

// Copy from and resize operation
at::Tensor _copy_from_and_resize_nkipy(const at::Tensor &self,
                                       const at::Tensor &dst);

// Fill operation
at::Tensor &fill_scalar_nkipy(at::Tensor &self, const at::Scalar &value);

} // namespace nkipy_tensor
