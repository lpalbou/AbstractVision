"""Facts about the host machine that backend-neutral code needs, each behind one function.

Each function returns None when the fact is not available on this platform, and callers treat
None as unknown. Platforms are added here, in one place, rather than in the backends.
"""

from __future__ import annotations

import os
from typing import Optional


def total_physical_memory_bytes() -> Optional[int]:
    """Physical memory of the machine, in bytes, or None when this platform does not report it.

    macOS and Linux report it through POSIX sysconf. Other platforms (Windows) are not
    implemented yet and return None.
    """
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return None
    try:
        total = int(sysconf("SC_PAGE_SIZE")) * int(sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, TypeError):
        return None
    return total if total > 0 else None
