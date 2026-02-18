# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime backend implementation using spiky's native C++ extension."""

import os

from spiky.backend.protocol import RuntimeBackend


class SpikyBackend(RuntimeBackend):
    """Runtime backend implementation using spiky's native C++ extension."""

    def __init__(self):
        self._initialized = False

    def register_torch_device(self) -> None:
        """Register nkipy as a PyTorch custom device."""
        # Register torch.compile backend ("nkipy") from spiky.torch.
        try:
            import spiky.torch  # noqa: F401
        except Exception:
            # Keep going: device registration can still happen via spike_torch.
            pass

        # Ensure PrivateUse1 device hooks/module are registered as "nkipy".
        # spiky.torch only registers the compile backend; the custom device
        # registration still comes from spike_torch.
        import spike_torch  # noqa: F401

    def init(self, visible_core: int) -> None:
        """Initialize spiky runtime for the given core."""
        import spiky

        # Set root comm ID for collectives
        if os.environ.get("NEURON_RT_ROOT_COMM_ID", None) is None:
            root_addr = os.environ.get("MASTER_ADDR", "localhost")
            root_port = os.environ.get("NEURON_RT_PORT", "61234")
            os.environ["NEURON_RT_ROOT_COMM_ID"] = f"{root_addr}:{root_port}"

        # Set visible cores via env var before init
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(visible_core)

        # Initialize spiky with device 0 (relative to visible cores)
        spiky.init(device_id=0)

        self._initialized = True

    def close(self) -> None:
        """Close spiky runtime."""
        import spiky

        spiky.close()

        self._initialized = False

    def is_initialized(self) -> bool:
        """Check if spiky runtime is initialized."""
        import spiky

        return spiky.is_initialized() and self._initialized

    def current_device(self) -> int:
        """Return current device index."""
        import spiky.torch

        return spiky.torch.current_device()

    def set_device(self, device: int) -> None:
        """Set current device."""
        import spiky.torch

        spiky.torch.set_device(device)

    def device_count(self) -> int:
        """Return number of available devices."""
        import spiky

        return spiky.device_count()
