"""Data preparation helpers used by training scripts."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def to_onehot(seg: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
    """Convert integer segmentation tensor to one-hot NCDHW float tensor.

    Accepts tensors shaped `(B, 1, H, W, D)` or `(B, H, W, D)`.
    """
    if seg.ndim == 5:
        if seg.shape[1] != 1:
            raise ValueError(f"Expected single-channel segmentation, got shape {tuple(seg.shape)}")
        seg = seg[:, 0]
    elif seg.ndim != 4:
        raise ValueError(f"Expected segmentation shape (B,1,H,W,D) or (B,H,W,D), got {tuple(seg.shape)}")

    seg = seg.long()
    if torch.any(seg < 0):
        raise ValueError("Segmentation labels must be non-negative")
    if num_classes is None:
        num_classes = int(torch.max(seg).item()) + 1
    if num_classes <= int(torch.max(seg).item()):
        raise ValueError(f"num_classes={num_classes} is too small for max label {int(torch.max(seg).item())}")

    return F.one_hot(seg, num_classes=num_classes).permute(0, 4, 1, 2, 3).contiguous().float()


def to_onehot_seg_o(seg: torch.Tensor) -> torch.Tensor:
    """One-hot encode OAR/soft-tissue labels.

    Set `MUSA_SEG_O_CLASSES` to force a fixed channel count across validation
    pairs if your prepared labels are not guaranteed to share the same max label.
    """
    return to_onehot(seg, num_classes=_env_int("MUSA_SEG_O_CLASSES"))


def to_onehot_seg_b(seg: torch.Tensor) -> torch.Tensor:
    """One-hot encode binary bone labels."""
    return to_onehot(seg, num_classes=_env_int("MUSA_SEG_B_CLASSES") or 2)


def max_label_in_folder(folder: str) -> int:
    """Return the maximum integer label among `.npy` segmentations in a folder."""
    paths = sorted(Path(folder).glob("*.npy"))
    if not paths:
        raise ValueError(f"No .npy files found in {folder}")
    max_label = 0
    for path in paths:
        array = np.load(path, mmap_mode="r")
        max_label = max(max_label, int(np.max(array)))
    return max_label
