// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace nkipy_device {

// Get the current Neuron device index
int current_device();

// Set the current Neuron device
void set_device(int device);

// Get device count
int device_count();

} // namespace nkipy_device
