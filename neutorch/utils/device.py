"""Device detection and tensor relocation helpers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Union

import torch
from torch import Tensor


def get_device(
    prefer_gpu: bool = True,
    *,
    allow_mps: bool = True,
) -> torch.device:
    """Select the best available compute device.

    Priority when ``prefer_gpu`` is ``True``:
        1. CUDA (if available)
        2. Apple MPS (if ``allow_mps`` and available)
        3. CPU

    Args:
        prefer_gpu: Prefer an accelerator over CPU when one exists.
        allow_mps: Include Apple Metal Performance Shaders as a
            candidate backend.

    Returns:
        A resolved ``torch.device``.
    """
    if prefer_gpu:
        if is_cuda_available():
            return torch.device("cuda")
        if allow_mps and is_mps_available():
            return torch.device("mps")
    return torch.device("cpu")


def to_device(
    obj: Any,
    device: Union[str, torch.device],
    *,
    non_blocking: bool = False,
) -> Any:
    """Recursively move tensors (and nested containers) to ``device``.

    Supported structures: ``Tensor``, ``dict``, ``list``, ``tuple``,
    and plain Python scalars / non-tensor objects (returned unchanged).

    Args:
        obj: Object graph potentially containing tensors.
        device: Target device string or ``torch.device``.
        non_blocking: Passed through to ``Tensor.to`` for pinned-memory
            async transfers.

    Returns:
        A new structure with all tensors relocated; non-tensors are
        returned as-is.
    """
    if isinstance(obj, Tensor):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, Mapping):
        return {
            key: to_device(value, device, non_blocking=non_blocking)
            for key, value in obj.items()
        }
    if isinstance(obj, tuple):
        return tuple(to_device(value, device, non_blocking=non_blocking) for value in obj)
    if isinstance(obj, list):
        return [to_device(value, device, non_blocking=non_blocking) for value in obj]
    return obj


def module_to_device(
    module: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    *,
    prefer_gpu: bool = True,
    allow_mps: bool = True,
) -> tuple[torch.nn.Module, torch.device]:
    """Move an ``nn.Module`` to a resolved device.

    Args:
        module: Module to relocate.
        device: Explicit target device. When ``None``,
            :func:`get_device` is used.
        prefer_gpu: Forwarded to :func:`get_device` when auto-selecting.
        allow_mps: Forwarded to :func:`get_device` when auto-selecting.

    Returns:
        Tuple of ``(module_on_device, resolved_device)``.
    """
    resolved = (
        torch.device(device)
        if device is not None
        else get_device(prefer_gpu=prefer_gpu, allow_mps=allow_mps)
    )
    return module.to(resolved), resolved


def is_cuda_available() -> bool:
    """Return whether CUDA is usable on this machine."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Return whether Apple MPS is usable on this machine."""
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def describe_device(device: Optional[Union[str, torch.device]] = None) -> str:
    """Human-readable summary of a device (name, capability, memory).

    Args:
        device: Device to describe. Defaults to :func:`get_device`.

    Returns:
        A short descriptive string suitable for logging.
    """
    resolved = torch.device(device) if device is not None else get_device()
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        total_mem_gb = torch.cuda.get_device_properties(index).total_memory / (1024**3)
        return f"cuda:{index} ({name}, {total_mem_gb:.1f} GiB)"
    if resolved.type == "mps":
        return "mps (Apple Metal)"
    return "cpu"


def batch_to_device(
    batch: Union[Tensor, Iterable[Any], dict[str, Any]],
    device: Union[str, torch.device],
    *,
    non_blocking: bool = True,
) -> Any:
    """Convenience wrapper for relocating a DataLoader batch.

    Equivalent to :func:`to_device` with ``non_blocking=True`` by
    default, matching typical training-loop usage with pinned memory.

    Args:
        batch: A DataLoader batch (tensor, tuple/list, or dict).
        device: Target device.
        non_blocking: Async transfer flag.

    Returns:
        Batch with tensors on ``device``.
    """
    return to_device(batch, device, non_blocking=non_blocking)
