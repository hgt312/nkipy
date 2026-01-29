// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Device.h>
#include <c10/core/Storage.h>
#include <c10/util/Optional.h>

namespace torch_nkipy {

// NKIPyHooksInterface implements the PrivateUse1HooksInterface to enable
// autograd support for NKIPy tensors. This interface is required for
// gradient computation and other autograd operations.
class NKIPyHooksInterface : public at::PrivateUse1HooksInterface {
public:
  // Constructor
  NKIPyHooksInterface() = default;

  // Destructor
  ~NKIPyHooksInterface() override = default;

  // Initialize the NKIPy runtime if needed
  void init() const override;

  // Check if a pointer is pinned memory (always true for NKIPy as all tensors
  // are pinned)
  bool isPinnedPtr(const void *data) const override { return true; }

  // Get the device index
  c10::DeviceIndex getDeviceIndex(c10::DeviceIndex device_index) const {
    return device_index;
  }

  // Check if NKIPy runtime is available / has primary context
  bool hasPrimaryContext(c10::DeviceIndex device_index) const override;

  // Allocate pinned memory using the NKIPy host allocator
  c10::Allocator *getPinnedMemoryAllocator() const override {
    return at::getHostAllocator(at::kPrivateUse1);
  }

  // Resize storage for PrivateUse1 (NKIPy) tensors
  void resizePrivateUse1Bytes(const c10::Storage &storage,
                              size_t new_bytes) const override;

  // Check if NKIPy runtime is available
  bool isAvailable() const;
};

// Get the singleton instance of NKIPyHooksInterface
at::PrivateUse1HooksInterface *get_nkipy_hooks();

// Explicitly register the hooks interface with PyTorch
// This can be called from Python to ensure registration happens
void register_nkipy_hooks();

} // namespace torch_nkipy
