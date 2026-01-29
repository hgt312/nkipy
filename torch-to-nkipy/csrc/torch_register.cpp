// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/CPUFallback.h>

#include "nkipy_tensor.h"
#include "torch_register.h"

namespace torch_nkipy {

// Custom CPU fallback with warning system
void nkipy_cpu_fallback(const c10::OperatorHandle &op,
                        torch::jit::Stack *stack) {
  auto op_name = c10::toString(op.schema().operator_name());
  std::cout << "[Warning] CPU fallback operator: " << op_name << std::endl;
  at::native::cpu_fallback(op, stack);
}

// Simple autograd fallback that redispatches to the actual implementation
// This prevents the CPU fallback warning for operations that have
// implementations
void autograd_fallback(const c10::OperatorHandle &op,
                       c10::DispatchKeySet dispatch_keys,
                       torch::jit::Stack *stack) {
  // Remove autograd keys and redispatch to the actual implementation
  auto new_keys = dispatch_keys & c10::after_autograd_keyset;
  op.redispatchBoxed(new_keys, stack);
}

// Register 'nkipy' as the name for PrivateUse1 device type
void register_nkipy_device() { c10::register_privateuse1_backend("nkipy"); }

} // namespace torch_nkipy

C10_REGISTER_GUARD_IMPL(PrivateUse1, torch_nkipy::NKIPyDeviceGuardImpl);

// Register CPU fallback for unimplemented operations
// This happens at module load time, but can be overridden by Python ops
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<
             &torch_nkipy::nkipy_cpu_fallback>());
}

// Register autograd fallback that just redispatches
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<
             &torch_nkipy::autograd_fallback>());
}

// Register native functions with PyTorch dispatch
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &nkipy_tensor::empty_memory_format_nkipy);
  m.impl("empty_strided", &nkipy_tensor::empty_strided_nkipy);
  m.impl("resize_", &nkipy_tensor::resize_nkipy);
  // Copy operations
  m.impl("copy_", &nkipy_tensor::copy_nkipy);
  m.impl("_to_copy", &nkipy_tensor::_to_copy_nkipy);
  m.impl("_copy_from_and_resize", &nkipy_tensor::_copy_from_and_resize_nkipy);
  // View operations
  m.impl("view", &nkipy_tensor::view_nkipy);
  m.impl("as_strided", &nkipy_tensor::as_strided_nkipy);
  m.impl("_reshape_alias", &nkipy_tensor::_reshape_alias_nkipy);
  // Fill operations
  m.impl("fill_.Scalar", &nkipy_tensor::fill_scalar_nkipy);
}
