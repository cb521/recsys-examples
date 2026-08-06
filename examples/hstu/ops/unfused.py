# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

FORCE_UNFUSED_HSTU_ENV = "HSTU_FORCE_UNFUSED"
LEGACY_FORCE_UNFUSED_ADDMM_SILU_ENV = "HSTU_FORCE_UNFUSED_ADDMM_SILU"

_TRUE_VALUES = ("1", "true", "yes", "on")


def should_force_unfused_hstu() -> bool:
    """Return whether HSTU benchmark paths should use ordinary PyTorch ops."""
    return any(
        os.environ.get(name, "0").lower() in _TRUE_VALUES
        for name in (
            FORCE_UNFUSED_HSTU_ENV,
            LEGACY_FORCE_UNFUSED_ADDMM_SILU_ENV,
        )
    )
