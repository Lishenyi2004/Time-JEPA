"""AgiBot adaptation for the official PSG-JEPA implementation.

The modules in this package contain only project-local data and training glue.
The PSG-JEPA model itself is imported from a separately pinned official clone.
"""

from .data import AgiBotPSGWindowDataset, collate_psg_windows
from .protocol import (
    EXPECTED_UPSTREAM_COMMIT,
    AgiBotPSGProtocol,
    AgiBotPSGRecord,
)
from .static_only import masked_mse, static_only_forward

__all__ = [
    "EXPECTED_UPSTREAM_COMMIT",
    "AgiBotPSGProtocol",
    "AgiBotPSGRecord",
    "AgiBotPSGWindowDataset",
    "collate_psg_windows",
    "masked_mse",
    "static_only_forward",
]

