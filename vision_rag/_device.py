"""
vision_rag/_device.py

Shared device-resolution utility for every torch-backed local component
(CLIP embedders, local Whisper). Gives users explicit control over device
placement ("auto" / "cpu" / "cuda" / "cuda:N" / "mps") with loud, specific
errors instead of silently falling back to CPU when a requested device
isn't actually available.

Internal module — not part of the public API in __init__.py.

Usage:
    from vision_rag._device import resolve_device

    device = resolve_device("auto")                     # best available
    device = resolve_device("cuda")                      # explicit, errors if unavailable
    device = resolve_device("cuda:1")
    device = resolve_device("mps")
    device = resolve_device("cpu")                       # always available, no torch needed

    # restrict to a subset of device kinds a given backend actually supports
    # (e.g. faster-whisper's ctranslate2 backend has no MPS support)
    device = resolve_device("auto", allowed={"cpu", "cuda"})
"""

from __future__ import annotations

import re

_VALID_PATTERN = re.compile(r"^(cpu|mps|cuda(:\d+)?)$")


def resolve_device(requested: str, allowed: set[str] | None = None) -> str:
    """
    Resolve a user-requested device string into a concrete, available device.

    Parameters
    ----------
    requested : str
        One of "auto", "cpu", "mps", "cuda", or "cuda:N".
    allowed : set[str] | None
        Optional set of device *kinds* ("cpu", "cuda", "mps") the calling
        backend actually supports. If the requested device's kind isn't in
        this set, a ValueError is raised (for an explicit request) or the
        kind is skipped when resolving "auto".

    Returns
    -------
    str — a concrete device string ready to hand to the underlying library.

    Raises
    ------
    ValueError
        If `requested` is not a recognized device string (checked first,
        before torch is ever imported -- so an unrecognized string like
        "tpu" is always reported as such, regardless of whether torch
        happens to be installed), or if it names a device kind excluded
        by `allowed`.
    RuntimeError
        If `requested` names a specific device (not "auto") that is not
        actually available on this machine (e.g. "cuda" with no GPU, or
        "mps" on non-Apple-Silicon hardware).
    """
    requested = requested.strip().lower()

    # 1. Validate the STRING SHAPE first, before ever touching torch.
    if requested != "auto" and not _VALID_PATTERN.match(requested):
        raise ValueError(
            f"Unrecognized device string: {requested!r}. "
            "Expected one of: 'auto', 'cpu', 'mps', 'cuda', 'cuda:N'."
        )

    kind = "cuda" if requested.startswith("cuda") else requested
    if requested != "auto" and allowed is not None and kind not in allowed:
        raise ValueError(
            f"Device kind {kind!r} is not supported by this component. "
            f"Allowed device kinds here: {sorted(allowed)}."
        )

    # 2. "cpu" never needs torch and is always available.
    if requested == "cpu":
        return "cpu"

    # 3. Anything else ("auto" / "cuda" / "cuda:N" / "mps") needs torch to
    #    check real availability.
    try:
        import torch
    except ImportError:
        if requested == "auto":
            # auto degrades gracefully to cpu when torch isn't installed
            return "cpu"
        raise RuntimeError(
            f"Device {requested!r} requires PyTorch. Install it with: pip install torch"
        )

    def _mps_available() -> bool:
        backend = getattr(torch.backends, "mps", None)
        return backend is not None and backend.is_available()

    if requested == "auto":
        if (allowed is None or "cuda" in allowed) and torch.cuda.is_available():
            return "cuda"
        if (allowed is None or "mps" in allowed) and _mps_available():
            return "mps"
        return "cpu"

    if requested == "mps":
        if not _mps_available():
            raise RuntimeError(
                "Requested device 'mps' but MPS is not available on this machine "
                "(requires Apple Silicon + macOS 12.3+ with a compatible PyTorch build)."
            )
        return "mps"

    # cuda or cuda:N
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {requested!r} but CUDA is not available on this machine."
        )
    if ":" in requested:
        idx = int(requested.split(":", 1)[1])
        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested device {requested!r} but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
    return requested