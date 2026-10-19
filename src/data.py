"""CIFAR-10 loading, the fixed train/validation split, and normalization.

Owner: Afnan (data pipeline).

Everything is kept as uint8 arrays (N, 32, 32, 3) until the very last step,
so the training images and the degraded test images go through exactly the
same conversion and normalization.
"""
from __future__ import annotations

import numpy as np
import torch

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


def load_cifar10(root: str = "data"):
    """Download (if needed) and return CIFAR-10 as uint8 numpy arrays.

    Returns (train_x, train_y, test_x, test_y) with x of shape (N, 32, 32, 3).
    """
    from torchvision.datasets import CIFAR10

    train = CIFAR10(root, train=True, download=True)
    test = CIFAR10(root, train=False, download=True)
    return (np.asarray(train.data, dtype=np.uint8), np.asarray(train.targets, dtype=np.int64),
            np.asarray(test.data, dtype=np.uint8), np.asarray(test.targets, dtype=np.int64))


def stratified_split(labels: np.ndarray, val_per_class: int = 500, seed: int = 42):
    """Hold out `val_per_class` images per class for validation.

    Returns (train_idx, val_idx). The same seed always gives the same split.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        val_idx.append(idx[:val_per_class])
        train_idx.append(idx[val_per_class:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(val_idx))


def check_split(train_idx: np.ndarray, val_idx: np.ndarray, labels: np.ndarray, n_total: int):
    """Verify the split: no overlap, nothing lost, and every class equally represented in validation.

    The 10,000 CIFAR-10 test images are a separate file and are never part of this split.
    Raises AssertionError if anything is wrong; returns a small summary dict otherwise.
    """
    assert len(np.intersect1d(train_idx, val_idx)) == 0, "train and validation overlap"
    assert len(train_idx) + len(val_idx) == n_total, "some training images are missing"
    per_class = np.bincount(labels[val_idx], minlength=10)
    assert per_class.min() == per_class.max(), "validation set is not stratified"
    return {"train": int(len(train_idx)), "val": int(len(val_idx)), "val_per_class": per_class.tolist(),
            "overlap": 0}


def channel_stats(x_uint8: np.ndarray):
    """Per-channel mean and std on the [0, 1] scale, computed on the training subset only."""
    x = x_uint8.reshape(-1, 3).astype(np.float64) / 255.0
    return x.mean(axis=0).astype(np.float32), x.std(axis=0).astype(np.float32)


def to_tensor(x_uint8: np.ndarray) -> torch.Tensor:
    """uint8 (N, H, W, C) numpy -> uint8 (N, C, H, W) torch tensor."""
    return torch.from_numpy(np.ascontiguousarray(x_uint8)).permute(0, 3, 1, 2).contiguous()


class Normalizer(torch.nn.Module):
    """Converts uint8 images to normalized float. Applied AFTER any degradation."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return (x_uint8.float() / 255.0 - self.mean) / self.std


def augment_batch(x: torch.Tensor, padding: int = 4, flip: bool = True,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """Random crop (zero padding) + horizontal flip, done on a whole batch at once.

    Works on uint8 or float tensors (N, C, H, W) on any device, and is seeded
    through `generator`, so training is reproducible.
    """
    n, c, h, w = x.shape
    dev = x.device
    if padding > 0:
        padded = torch.nn.functional.pad(x, (padding,) * 4)
        oy = torch.randint(0, 2 * padding + 1, (n,), device=dev, generator=generator)
        ox = torch.randint(0, 2 * padding + 1, (n,), device=dev, generator=generator)
        iy = (oy[:, None] + torch.arange(h, device=dev)[None, :])          # (n, h)
        ix = (ox[:, None] + torch.arange(w, device=dev)[None, :])          # (n, w)
        b = torch.arange(n, device=dev)[:, None, None]
        x = padded.permute(0, 2, 3, 1)[b, iy[:, :, None], ix[:, None, :]].permute(0, 3, 1, 2)
    if flip:
        mask = torch.rand(n, device=dev, generator=generator) < 0.5
        x = torch.where(mask[:, None, None, None], x.flip(3), x)
    return x.contiguous()


def iterate_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool,
                    generator: torch.Generator | None = None):
    """Yield (x, y) mini-batches from tensors that already live on the target device."""
    n = x.shape[0]
    order = (torch.randperm(n, device=x.device, generator=generator) if shuffle
             else torch.arange(n, device=x.device))
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        yield x[idx], y[idx]
