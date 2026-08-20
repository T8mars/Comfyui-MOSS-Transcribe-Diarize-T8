from __future__ import annotations

import math


def estimate_max_new_tokens(duration_seconds: float) -> int:
    """Estimate a bounded transcript budget; generation may still stop at EOS."""
    return max(2048, min(65536, int(math.ceil(max(0.0, duration_seconds) * 16.0)) + 1024))


__all__ = ["estimate_max_new_tokens"]
