// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "include/spike_hooks.h"

#include <ATen/detail/PrivateUse1HooksInterface.h>

#include "include/spike_device.h"

namespace spike_torch {

void SpikeHooksInterface::init() const {
  // Spike runtime initialization is handled elsewhere (Python-side via nrt_init)
  // This is called by PyTorch when needed
}

bool SpikeHooksInterface::hasPrimaryContext(
    c10::DeviceIndex device_index) const {
  return device_index >= 0 &&
         device_index < static_cast<c10::DeviceIndex>(device::device_count());
}

void SpikeHooksInterface::resizePrivateUse1Bytes(const c10::Storage &storage,
                                                 size_t new_bytes) const {
  auto *alloc = storage.allocator();
  TORCH_CHECK(alloc != nullptr, "Storage must have an allocator");

  auto old_bytes = storage.nbytes();
  if (new_bytes == old_bytes) {
    return;
  }

  at::DataPtr new_data;
  if (new_bytes > 0) {
    new_data = alloc->allocate(new_bytes);
  }

  if (old_bytes > 0 && new_bytes > 0) {
    size_t copy_bytes = std::min(old_bytes, new_bytes);
    memcpy(new_data.get(), storage.data(), copy_bytes);
  }

  storage.set_data_ptr_noswap(std::move(new_data));
  storage.set_nbytes(new_bytes);
}

bool SpikeHooksInterface::isAvailable() const {
  return device::device_count() > 0;
}

// Singleton instance
static SpikeHooksInterface spike_hooks_instance;

at::PrivateUse1HooksInterface *get_spike_hooks() {
  return &spike_hooks_instance;
}

static bool hooks_registered = false;

void register_spike_hooks() {
  if (!hooks_registered) {
    at::RegisterPrivateUse1HooksInterface(get_spike_hooks());
    hooks_registered = true;
  }
}

} // namespace spike_torch
