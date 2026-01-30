// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "include/spike_allocator.h"

#include <c10/core/DeviceType.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <vector>

#include "include/spike_device.h"

extern "C" {
#include <nrt/nrt.h>
}

namespace spike_torch {
namespace allocator {

namespace {

// Global registry to track allocations (DataPtr deleter cannot capture state)
struct AllocatorRegistry {
  struct AllocationInfo {
    nrt_tensor_t *tensor;
    size_t size;
    int device;
  };

  std::unordered_map<void *, AllocationInfo> allocations;
  std::mutex mutex;

  static AllocatorRegistry &getInstance() {
    static AllocatorRegistry instance;
    return instance;
  }

  void add(void *ptr, nrt_tensor_t *tensor, size_t size, int device) {
    std::lock_guard<std::mutex> lock(mutex);
    allocations[ptr] = {tensor, size, device};
  }

  AllocationInfo remove(void *ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocations.find(ptr);
    if (it != allocations.end()) {
      AllocationInfo info = it->second;
      allocations.erase(it);
      return info;
    }
    return {nullptr, 0, -1};
  }

  nrt_tensor_t *find(void *ptr) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocations.find(ptr);
    return (it != allocations.end()) ? it->second.tensor : nullptr;
  }
};

// Simple memory pool that caches allocated blocks for reuse
class SimpleMemoryPool {
  struct Block {
    nrt_tensor_t *tensor;
    void *data_ptr;
    size_t size;
    std::chrono::steady_clock::time_point last_used;

    Block(nrt_tensor_t *t, void *ptr, size_t s)
        : tensor(t), data_ptr(ptr), size(s),
          last_used(std::chrono::steady_clock::now()) {}
  };

  std::vector<Block> blocks_;
  mutable std::mutex mutex_;

public:
  // Try to find a cached block of exact size
  std::pair<nrt_tensor_t *, void *> get_block(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Find exact size match
    auto it = std::find_if(blocks_.begin(), blocks_.end(),
                           [size](const Block &b) { return b.size == size; });

    if (it != blocks_.end()) {
      // Found a match - remove from cache and return
      nrt_tensor_t *tensor = it->tensor;
      void *data_ptr = it->data_ptr;
      blocks_.erase(it);
      return {tensor, data_ptr};
    }

    return {nullptr, nullptr};
  }

  // Add a freed block to the cache
  void add_block(nrt_tensor_t *tensor, void *data_ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_.emplace_back(tensor, data_ptr, size);
  }

  // Clear all cached blocks
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &block : blocks_) {
      nrt_tensor_free(&block.tensor);
    }
    blocks_.clear();
  }

  size_t cached_blocks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return blocks_.size();
  }
};

// Per-device memory pools
std::vector<SimpleMemoryPool> *memory_pools = nullptr;
std::mutex pools_init_mutex;

// Get or initialize the memory pools
std::vector<SimpleMemoryPool> &getMemoryPools() {
  std::lock_guard<std::mutex> lock(pools_init_mutex);
  if (!memory_pools) {
    int count = device::device_count();
    if (count <= 0) {
      throw std::runtime_error(
          "No Neuron devices available. Cannot initialize memory pools.");
    }
    memory_pools = new std::vector<SimpleMemoryPool>(count);
  }
  return *memory_pools;
}

// Get pool for specific device
SimpleMemoryPool &getDevicePool(int device) {
  auto &pools = getMemoryPools();
  if (device < 0 || device >= static_cast<int>(pools.size())) {
    throw std::runtime_error("Invalid device index: " + std::to_string(device));
  }
  return pools[device];
}

// RAII helper to temporarily suppress stderr
class StderrSuppressor {
  int saved_stderr;
  int devnull;

public:
  explicit StderrSuppressor(bool should_suppress)
      : saved_stderr(-1), devnull(-1) {
    if (!should_suppress)
      return;

    saved_stderr = dup(STDERR_FILENO);
    if (saved_stderr == -1)
      return;

    devnull = open("/dev/null", O_WRONLY);
    if (devnull == -1) {
      close(saved_stderr);
      saved_stderr = -1;
      return;
    }

    dup2(devnull, STDERR_FILENO);
  }

  ~StderrSuppressor() {
    if (saved_stderr != -1) {
      dup2(saved_stderr, STDERR_FILENO);
      close(saved_stderr);
    }
    if (devnull != -1) {
      close(devnull);
    }
  }

  StderrSuppressor(const StderrSuppressor &) = delete;
  StderrSuppressor &operator=(const StderrSuppressor &) = delete;
  StderrSuppressor(StderrSuppressor &&) = delete;
  StderrSuppressor &operator=(StderrSuppressor &&) = delete;
};

// Static deleter function for DataPtr
void spike_deleter(void *ctx) {
  nrt_tensor_t *t = static_cast<nrt_tensor_t *>(ctx);

  void *ptr = nrt_tensor_get_va(t);
  if (!ptr) {
    nrt_tensor_free(&t);
    return;
  }

  auto info = AllocatorRegistry::getInstance().remove(ptr);

  if (info.tensor && info.size > 0 && info.device >= 0) {
    // Add to device-specific cache instead of freeing
    auto &device_pool = getDevicePool(info.device);
    device_pool.add_block(info.tensor, ptr, info.size);
  } else {
    throw std::runtime_error(
        "SpikeAllocator: Deleter called on unregistered pointer. "
        "This indicates a bug - either double-free or missing registration.");
  }
}

class SpikeAllocator : public c10::Allocator {
public:
  SpikeAllocator() = default;

  c10::DataPtr allocate(size_t size) override {
    // Handle zero-size allocation
    if (size == 0) {
      return c10::DataPtr(nullptr,
                          c10::Device(c10::DeviceType::PrivateUse1, 0));
    }

    auto dev = device::current_device();

    // First check the device-specific cache
    auto &device_pool = getDevicePool(dev);
    auto [cached_tensor, cached_ptr] = device_pool.get_block(size);
    if (cached_tensor != nullptr) {
      AllocatorRegistry::getInstance().add(cached_ptr, cached_tensor, size,
                                           dev);
      return c10::DataPtr(cached_ptr, cached_tensor, &spike_deleter,
                          c10::Device(c10::DeviceType::PrivateUse1, dev));
    }

    // No cached block - allocate new
    nrt_tensor_t *tensor = nullptr;
    NRT_STATUS status;

    // Suppress errors on first attempt if we have cached blocks
    {
      StderrSuppressor suppressor(device_pool.cached_blocks() > 0);
      status = nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, dev, size,
                                   nullptr, &tensor);
    }

    // Handle OOM by clearing cache
    if (status == NRT_RESOURCE && device_pool.cached_blocks() > 0) {
      device_pool.clear();
      status = nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, dev, size,
                                   nullptr, &tensor);
    }

    if (status != NRT_SUCCESS || !tensor) {
      std::string error_msg = "Failed to allocate Neuron tensor of size " +
                              std::to_string(size) + " bytes";
      if (status == NRT_RESOURCE) {
        error_msg += " (out of memory)";
      }
      error_msg += ". Status: " + std::to_string(status);
      throw std::runtime_error(error_msg);
    }

    void *data = nrt_tensor_get_va(tensor);
    if (!data) {
      nrt_tensor_free(&tensor);
      throw std::runtime_error("Failed to get tensor data pointer");
    }

    AllocatorRegistry::getInstance().add(data, tensor, size, dev);

    return c10::DataPtr(data, tensor, &spike_deleter,
                        c10::Device(c10::DeviceType::PrivateUse1, dev));
  }

  void copy_data(void *dest, const void *src, std::size_t size) const override {
    auto &registry = AllocatorRegistry::getInstance();

    nrt_tensor_t *src_tensor = registry.find(const_cast<void *>(src));
    nrt_tensor_t *dst_tensor = registry.find(dest);

    if (src_tensor && dst_tensor) {
      NRT_STATUS status =
          nrt_tensor_copy(src_tensor, 0, dst_tensor, 0, size);
      if (status != NRT_SUCCESS) {
        throw std::runtime_error("nrt_tensor_copy failed");
      }
    } else {
      throw std::runtime_error(
          "SpikeAllocator::copy_data called with unknown pointers. "
          "Both source and destination must be allocated by this allocator.");
    }
  }
};

// Global singleton instance
std::once_flag allocator_init_flag;
SpikeAllocator *allocator_instance = nullptr;

SpikeAllocator *getAllocator() {
  std::call_once(allocator_init_flag,
                 []() { allocator_instance = new SpikeAllocator(); });
  return allocator_instance;
}

} // anonymous namespace

c10::Allocator *get() { return getAllocator(); }

void empty_cache() {
  auto &pools = getMemoryPools();
  for (auto &pool : pools) {
    pool.clear();
  }
}

size_t get_cached_blocks() {
  size_t total = 0;
  auto &pools = getMemoryPools();
  for (const auto &pool : pools) {
    total += pool.cached_blocks();
  }
  return total;
}

bool copy_tensor_data(void *dest, const void *src, size_t size) {
  auto &registry = AllocatorRegistry::getInstance();

  nrt_tensor_t *src_tensor = registry.find(const_cast<void *>(src));
  nrt_tensor_t *dst_tensor = registry.find(dest);

  if (src_tensor && dst_tensor) {
    NRT_STATUS status = nrt_tensor_copy(src_tensor, 0, dst_tensor, 0, size);
    return status == NRT_SUCCESS;
  }
  return false;
}

nrt_tensor_t *find_tensor(void *ptr) {
  return AllocatorRegistry::getInstance().find(ptr);
}

nrt_tensor_t *get_tensor_from_context(void *ctx) {
  return static_cast<nrt_tensor_t *>(ctx);
}

} // namespace allocator
} // namespace spike_torch
