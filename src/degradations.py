"""The four degradations, the 13 test conditions, and the severity check (PSNR/SSIM).

Owner: Afnan (degraded test sets).

All functions take and return uint8 arrays of shape (N, 32, 32, 3), so every
model receives exactly the same stored images.
"""
from __future__ import annotations

import hashlib
import io
import json
import os

import numpy as np
import torch
from PIL import Image

# (name, degradation type, severity level, parameters) -- matches Table III of the report.
CONDITIONS = [
    ("clean", "clean", 0, {}),
    ("blur_1", "blur", 1, {"sigma": 0.5, "kernel": 5}),
    ("blur_2", "blur", 2, {"sigma": 1.0, "kernel": 7}),
    ("blur_3", "blur", 3, {"sigma": 1.5, "kernel": 11}),
    ("noise_1", "noise", 1, {"std": 0.04}),
    ("noise_2", "noise", 2, {"std": 0.08}),
    ("noise_3", "noise", 3, {"std": 0.12}),
    ("lowres_1", "lowres", 1, {"size": 24}),
    ("lowres_2", "lowres", 2, {"size": 16}),
    ("lowres_3", "lowres", 3, {"size": 8}),
    ("jpeg_1", "jpeg", 1, {"quality": 60}),
    ("jpeg_2", "jpeg", 2, {"quality": 30}),
    ("jpeg_3", "jpeg", 3, {"quality": 10}),
]
DEGRADATION_TYPES = ("blur", "noise", "lowres", "jpeg")


def gaussian_blur(x: np.ndarray, sigma: float, kernel: int) -> np.ndarray:
    """Gaussian blur with a square kernel (reflect padding at the borders)."""
    from torchvision.transforms.functional import gaussian_blur as tv_blur

    t = torch.from_numpy(x).permute(0, 3, 1, 2).float()
    out = tv_blur(t, kernel_size=[kernel, kernel], sigma=[sigma, sigma])
    return out.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


def gaussian_noise(x: np.ndarray, std: float, seed: int) -> np.ndarray:
    """Additive Gaussian noise with standard deviation `std` on the [0, 1] pixel scale."""
    rng = np.random.default_rng(seed)
    noisy = x.astype(np.float64) / 255.0 + rng.normal(0.0, std, size=x.shape)
    return np.clip(np.round(noisy * 255.0), 0, 255).astype(np.uint8)


def reduce_resolution(x: np.ndarray, size: int) -> np.ndarray:
    """Downsample to size x size, then upsample back to the original size (bicubic both ways)."""
    h, w = x.shape[1:3]
    out = np.empty_like(x)
    for i, img in enumerate(x):
        small = Image.fromarray(img).resize((size, size), Image.BICUBIC)
        out[i] = np.asarray(small.resize((w, h), Image.BICUBIC))
    return out


def jpeg_compress(x: np.ndarray, quality: int) -> np.ndarray:
    """Encode and decode each image as JPEG at the given quality (4:2:0 chroma subsampling)."""
    out = np.empty_like(x)
    for i, img in enumerate(x):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=quality, subsampling=2)
        buf.seek(0)
        out[i] = np.asarray(Image.open(buf).convert("RGB"))
    return out


def apply_condition(x: np.ndarray, name: str, seed: int = 1234) -> np.ndarray:
    """Apply one of the 13 named conditions to a batch of uint8 images."""
    spec = {c[0]: c for c in CONDITIONS}[name]
    _, kind, level, p = spec
    if kind == "clean":
        return x.copy()
    if kind == "blur":
        return gaussian_blur(x, p["sigma"], p["kernel"])
    if kind == "noise":
        return gaussian_noise(x, p["std"], seed + level)   # a different, fixed seed per level
    if kind == "lowres":
        return reduce_resolution(x, p["size"])
    if kind == "jpeg":
        return jpeg_compress(x, p["quality"])
    raise ValueError(name)


def distortion_stats(clean: np.ndarray, degraded: np.ndarray):
    """Mean PSNR (dB) and mean SSIM between clean and degraded images."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    psnr, ssim = [], []
    for a, b in zip(clean, degraded):
        if np.array_equal(a, b):
            psnr.append(float("inf")); ssim.append(1.0); continue
        psnr.append(peak_signal_noise_ratio(a, b, data_range=255))
        ssim.append(structural_similarity(a, b, channel_axis=-1, data_range=255))
    psnr = np.array(psnr)
    finite = psnr[np.isfinite(psnr)]
    return (float(finite.mean()) if finite.size else float("inf")), float(np.mean(ssim))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_test_sets(test_x: np.ndarray, test_y: np.ndarray, out_dir: str, seed: int = 1234):
    """Create all 13 test sets once, save them, and write a manifest with checksums and PSNR/SSIM."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"seed": seed, "n_images": int(len(test_x)), "conditions": []}
    for name, kind, level, params in CONDITIONS:
        xd = apply_condition(test_x, name, seed)
        path = os.path.join(out_dir, f"{name}.npz")
        np.savez_compressed(path, x=xd, y=test_y)
        psnr, ssim = distortion_stats(test_x, xd)
        manifest["conditions"].append({
            "name": name, "type": kind, "level": level, "params": params,
            "psnr_db": psnr, "ssim": ssim, "sha256": sha256(path)})
        print(f"{name:9s} PSNR={psnr:6.2f} dB  SSIM={ssim:.3f}")
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_condition(degraded_dir: str, name: str):
    d = np.load(os.path.join(degraded_dir, f"{name}.npz"))
    return d["x"], d["y"]
