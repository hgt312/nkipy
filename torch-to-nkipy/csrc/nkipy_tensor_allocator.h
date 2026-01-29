// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <c10/core/Allocator.h>

// Forward declaration for nrt_tensor
struct nrt_tensor;
typedef struct nrt_tensor nrt_tensor_t;

namespace nkipy_tensor_allocator {

// Get the singleton allocator instance
c10::Allocator *get();

// Empty the cache (no-op for now since we don't have caching yet)
void emptyCache();

// Get cached blocks count
size_t getCachedBlocks();

// Copy data between neuron tensors
// Returns true if successful, false if tensors not found
bool copyTensorData(void *dest, const void *src, size_t size);

// Find the nrt_tensor for a given data pointer
// Returns the nrt_tensor_t* if found, nullptr otherwise
nrt_tensor_t *findTensor(void *ptr);

} // namespace nkipy_tensor_allocator
