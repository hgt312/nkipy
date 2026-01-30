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

#include "include/spike_allocator.h"
#include "include/spike_device.h"
#include "include/spike_hooks.h"
#include "include/spike_tensor_impl.h"
#include "include/spike_torch.h"

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "spike-torch: PyTorch device integration for spike runtime";

  // Device registration
  m.def("_register_spike_device", &spike_torch::register_spike_device,
        "Register spike as a device type in PyTorch");

  m.def("_register_spike_hooks", &spike_torch::register_spike_hooks,
        "Register PrivateUse1HooksInterface for spike device");

  // Device functions
  m.def(
      "_device_count",
      []() -> int64_t {
        return static_cast<int64_t>(spike_torch::device::device_count());
      },
      "Get the number of spike devices");

  m.def(
      "_current_device",
      []() -> int64_t {
        return static_cast<int64_t>(spike_torch::device::current_device());
      },
      "Get the current spike device");

  m.def(
      "_set_device",
      [](int64_t device) { spike_torch::device::set_device(device); },
      "Set the current spike device", py::arg("device"));

  m.def(
      "_is_available",
      []() -> bool { return spike_torch::device::is_available(); },
      "Check if spike devices are available");

  // Memory management
  m.def(
      "_empty_cache", []() { spike_torch::allocator::empty_cache(); },
      "Empty the spike memory cache");

  m.def(
      "_get_cached_blocks",
      []() -> int64_t {
        return static_cast<int64_t>(spike_torch::allocator::get_cached_blocks());
      },
      "Get the number of cached memory blocks");

  // NRT initialization
  m.def(
      "_nrt_init",
      []() {
        nrt_init(NRT_FRAMEWORK_TYPE_PYTORCH, TORCH_VERSION, "1.0");
        std::atexit([]() { nrt_close(); });
      },
      "Initialize Neuron runtime");

  m.def(
      "_nrt_close", []() { nrt_close(); },
      "Close Neuron runtime and clean up resources");

  // Model loading
  m.def(
      "_nrt_load",
      [](py::bytes neff_bytes) -> py::object {
        std::string neff_data = neff_bytes;
        nrt_model_t *model = nullptr;

        int current_device = spike_torch::device::current_device();

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

        int current_device = spike_torch::device::current_device();

        NRT_STATUS status = nrt_load_collectives(
            neff_data.data(), neff_data.size(), current_device, 1, device_id,
            device_count, &model);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to load NEFF. Status: " +
                                   std::to_string(status));
        }
        return py::int_(reinterpret_cast<uintptr_t>(model));
      },
      "Load a NEFF with collectives and return model handle",
      py::arg("neff_bytes"), py::arg("device_id"), py::arg("device_count"));

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

  // Tensor set management
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

        torch::Tensor torch_tensor = THPVariable_Unpack(tensor_obj.ptr());

        if (torch_tensor.device().type() != c10::DeviceType::PrivateUse1) {
          throw std::runtime_error("Tensor must be on spike device, but got: " +
                                   torch_tensor.device().str());
        }

        void *data_ptr = torch_tensor.data_ptr();
        size_t storage_offset_bytes =
            torch_tensor.storage_offset() * torch_tensor.element_size();
        void *base_ptr = static_cast<char *>(data_ptr) - storage_offset_bytes;

        nrt_tensor_t *nrt_tensor = spike_torch::allocator::find_tensor(base_ptr);
        if (!nrt_tensor) {
          throw std::runtime_error(
              "Could not find NRT tensor for PyTorch tensor. "
              "Ensure tensor is on spike device.");
        }

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

  // Model execution
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

  // Profiling
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
      "Collect results and disable profiling", py::arg("filename"));

  // Barrier
  m.def(
      "_nrt_barrier",
      [](py::int_ device_id, py::int_ global_device_id,
         py::int_ global_device_count) {
        NRT_STATUS status =
            nrt_barrier(device_id, global_device_id, global_device_count);
        if (status != NRT_SUCCESS) {
          throw std::runtime_error("Failed to execute barrier. Status: " +
                                   std::to_string(status));
        }
      },
      "Execute barrier across all devices", py::arg("device_id"),
      py::arg("global_device_id"), py::arg("global_device_count"));

  // Get NRT tensor from PyTorch tensor
  m.def(
      "_get_nrt_tensor",
      [](py::object tensor_obj) -> py::object {
        torch::Tensor torch_tensor = THPVariable_Unpack(tensor_obj.ptr());

        if (torch_tensor.device().type() != c10::DeviceType::PrivateUse1) {
          throw std::runtime_error("Tensor must be on spike device, but got: " +
                                   torch_tensor.device().str());
        }

        nrt_tensor_t *nrt_tensor = spike_torch::get_nrt_tensor(torch_tensor);
        if (!nrt_tensor) {
          return py::none();
        }
        return py::int_(reinterpret_cast<uintptr_t>(nrt_tensor));
      },
      "Get NRT tensor handle from PyTorch tensor", py::arg("tensor"));
}
