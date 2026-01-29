// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nkipy_hooks.h"

#include <ATen/detail/PrivateUse1HooksInterface.h>

#include "nkipy_device.h"

namespace torch_nkipy {

// Initialize the NKIPy runtime
void NKIPyHooksInterface::init() const {
  // NKIPy runtime initialization is handled elsewhere (Python-side via nrt_init)
  // This is called by PyTorch when needed
}

// Check if NKIPy has a primary context for the given device
bool NKIPyHooksInterface::hasPrimaryContext(
    c10::DeviceIndex device_index) const {
  // Return true if NKIPy is available and the device is valid
  return device_index >= 0 &&
         device_index < static_cast<c10::DeviceIndex>(nkipy_device::device_count());
}

// Resize storage for NKIPy tensors
void NKIPyHooksInterface::resizePrivateUse1Bytes(const c10::Storage &storage,
                                                  size_t new_bytes) const {
  // Get the allocator and resize
  auto *allocator = storage.allocator();
  TORCH_CHECK(allocator != nullptr, "Storage must have an allocator");

  auto old_bytes = storage.nbytes();
  if (new_bytes == old_bytes) {
    return;
  }

  // For NKIPy, we use the CPU allocator backend
  // Allocate new storage
  at::DataPtr new_data;
  if (new_bytes > 0) {
    new_data = allocator->allocate(new_bytes);
  }

  // Copy old data if needed
  if (old_bytes > 0 && new_bytes > 0) {
    size_t copy_bytes = std::min(old_bytes, new_bytes);
    memcpy(new_data.get(), storage.data(), copy_bytes);
  }

  // Swap the data
  storage.set_data_ptr_noswap(std::move(new_data));
  storage.set_nbytes(new_bytes);
}

// Check if NKIPy runtime is available
bool NKIPyHooksInterface::isAvailable() const {
  return nkipy_device::device_count() > 0;
}

// Singleton instance
static NKIPyHooksInterface nkipy_hooks_instance;

at::PrivateUse1HooksInterface *get_nkipy_hooks() {
  return &nkipy_hooks_instance;
}

// Flag to track if hooks have been registered
static bool hooks_registered = false;

// Explicitly register the hooks interface with PyTorch
void register_nkipy_hooks() {
  if (!hooks_registered) {
    at::RegisterPrivateUse1HooksInterface(get_nkipy_hooks());
    hooks_registered = true;
  }
}

} // namespace torch_nkipy
