// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <c10/core/DeviceType.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/extension.h>

#include <cstdlib>
#include <string>

#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_profile.h>

#include "nkipy_device.h"
#include "nkipy_hooks.h"
#include "nkipy_tensor_allocator.h"
#include "torch_register.h"

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_register_nkipy_device", &torch_nkipy::register_nkipy_device,
        "Register nkipy as a device type in PyTorch");

  m.def("_register_nkipy_hooks", &torch_nkipy::register_nkipy_hooks,
        "Register PrivateUse1HooksInterface for nkipy device");

  m.def(
      "_nrt_init",
      []() {
        // Initialize Neuron Runtime
        nrt_init(NRT_FRAMEWORK_TYPE_PYTORCH, // Framework type
                 "2.0", // Framework version (PyTorch version)
                 "1.0"  // FAL version
        );

        // Register cleanup function
        std::atexit([]() { nrt_close(); });
      },
      "Initialize Neuron runtime");

  m.def(
      "_nrt_close", []() { nrt_close(); },
      "Close Neuron runtime and clean up resources");

  m.def(
      "_nrt_device_count",
      []() -> int64_t {
        return static_cast<int64_t>(nkipy_device::device_count());
      },
      "Get the number of Neuron devices");

  // NRT functions for NKI kernel execution
  m.def(
      "_nrt_load",
      [](py::bytes neff_bytes) -> py::object {
        std::string neff_data = neff_bytes;
        nrt_model_t *model = nullptr;

        // Get current device to ensure model is loaded on the same core as
        // tensors
        int current_device = nkipy_device::current_device();

        // Load model on the current device
        NRT_STATUS status = nrt_load(neff_data.data(), neff_data.size(),
                                     current_device, 1, &model);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to load NEFF. Status: " +
                                   std::to_string(status));
        }
        return py::int_(reinterpret_cast<uintptr_t>(model));
      },
      "Load a NEFF and return model handle", py::arg("neff_bytes"));

  m.def(
      "_nrt_load_collectives",
      [](py::bytes neff_bytes, py::int_ device_id,
         py::int_ device_count) -> py::object {
        std::string neff_data = neff_bytes;
        nrt_model_t *model = nullptr;

        // Get current device to ensure model is loaded on the same core as
        // tensors
        int current_device = nkipy_device::current_device();

        // Load model on the current device
        NRT_STATUS status = nrt_load_collectives(
            neff_data.data(), neff_data.size(), current_device, 1, device_id,
            device_count, &model);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to load NEFF. Status: " +
                                   std::to_string(status));
        }
        return py::int_(reinterpret_cast<uintptr_t>(model));
      },
      "Load a NEFF and return model handle", py::arg("neff_bytes"),
      py::arg("device_id"), py::arg("device_count"));

  m.def(
      "_nrt_unload",
      [](py::int_ model_handle) {
        nrt_model_t *model =
            reinterpret_cast<nrt_model_t *>(model_handle.cast<uintptr_t>());
        NRT_STATUS status = nrt_unload(model);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to unload model. Status: " +
                                   std::to_string(status));
        }
      },
      "Unload a model", py::arg("model_handle"));

  m.def(
      "_nrt_allocate_tensor_set",
      []() -> py::object {
        nrt_tensor_set_t *tensor_set = nullptr;
        NRT_STATUS status = nrt_allocate_tensor_set(&tensor_set);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to allocate tensor set. Status: " +
                                   std::to_string(status));
        }
        return py::int_(reinterpret_cast<uintptr_t>(tensor_set));
      },
      "Allocate a new tensor set");

  m.def(
      "_nrt_destroy_tensor_set",
      [](py::int_ tensor_set_handle) {
        nrt_tensor_set_t *tensor_set = reinterpret_cast<nrt_tensor_set_t *>(
            tensor_set_handle.cast<uintptr_t>());
        nrt_destroy_tensor_set(&tensor_set);
      },
      "Destroy a tensor set", py::arg("tensor_set_handle"));

  m.def(
      "_nrt_add_tensor_to_tensor_set",
      [](py::int_ tensor_set_handle, py::object tensor_obj,
         const std::string &tensor_name) {
        nrt_tensor_set_t *tensor_set = reinterpret_cast<nrt_tensor_set_t *>(
            tensor_set_handle.cast<uintptr_t>());

        // Get PyTorch tensor using THPVariable_Unpack
        torch::Tensor torch_tensor = THPVariable_Unpack(tensor_obj.ptr());

        // Verify tensor is on nkipy device
        if (torch_tensor.device().type() != c10::DeviceType::PrivateUse1) {
          throw std::runtime_error(
              "Tensor must be on nkipy device, but got: " +
              torch_tensor.device().str());
        }

        // Get data pointer - adjust for storage offset to get base storage
        // pointer
        void *data_ptr = torch_tensor.data_ptr();
        size_t storage_offset_bytes =
            torch_tensor.storage_offset() * torch_tensor.element_size();
        void *base_ptr = static_cast<char *>(data_ptr) - storage_offset_bytes;

        // Find the corresponding nrt_tensor_t using the allocator with base
        // pointer
        nrt_tensor_t *nrt_tensor = nkipy_tensor_allocator::findTensor(base_ptr);
        if (!nrt_tensor) {
          throw std::runtime_error(
              "Could not find NRT tensor for PyTorch tensor. Ensure tensor is "
              "on nkipy device.");
        }

        // Add to tensor set
        NRT_STATUS status = nrt_add_tensor_to_tensor_set(
            tensor_set, tensor_name.c_str(), nrt_tensor);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error(
              "Failed to add tensor to tensor set. Status: " +
              std::to_string(status));
        }
      },
      "Add a PyTorch tensor to tensor set", py::arg("tensor_set_handle"),
      py::arg("tensor"), py::arg("tensor_name"));

  m.def(
      "_nrt_execute",
      [](py::int_ model_handle, py::int_ input_set_handle,
         py::int_ output_set_handle) {
        nrt_model_t *model =
            reinterpret_cast<nrt_model_t *>(model_handle.cast<uintptr_t>());
        nrt_tensor_set_t *input_set = reinterpret_cast<nrt_tensor_set_t *>(
            input_set_handle.cast<uintptr_t>());
        nrt_tensor_set_t *output_set = reinterpret_cast<nrt_tensor_set_t *>(
            output_set_handle.cast<uintptr_t>());

        NRT_STATUS status = nrt_execute(model, input_set, output_set);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to execute model. Status: " +
                                   std::to_string(status));
        }
      },
      "Execute a model", py::arg("model_handle"), py::arg("input_set_handle"),
      py::arg("output_set_handle"));

  m.def(
      "_nrt_profile_start",
      [](py::int_ model_handle, const std::string &filename) {
        nrt_model_t *model =
            reinterpret_cast<nrt_model_t *>(model_handle.cast<uintptr_t>());
        NRT_STATUS status = nrt_profile_start(model, filename.c_str());
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to start profiling. Status: " +
                                   std::to_string(status));
        }
      },
      "Enable profiling for a model", py::arg("model_handle"),
      py::arg("filename"));

  m.def(
      "_nrt_profile_stop",
      [](const std::string &filename) {
        NRT_STATUS status = nrt_profile_stop(filename.c_str());
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to stop profiling. Status: " +
                                   std::to_string(status));
        }
      },
      "Collect results and disable profiling for a model", py::arg("filename"));

  m.def(
      "_nrt_barrier",
      [](py::int_ device_id, py::int_ global_device_id,
         py::int_ global_device_count) {
        NRT_STATUS status =
            nrt_barrier(device_id, global_device_id, global_device_count);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to execute the device barrier " +
                                   std::to_string(status));
        }
      },
      "Execute barrier across all the devices", py::arg("device_id"),
      py::arg("global_device_id"), py::arg("global_device_count"));
}
