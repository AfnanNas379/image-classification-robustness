"""Train one model with the shared recipe.

Owner: Lulu (training).

Usage:
    python -m src.train --model resnet18 --seed 0 --out runs
Resumes automatically from runs/<model>_seed<seed>/last.pt if Colab disconnects.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from src import data as D
from src.models import build_model, count_parameters


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lr_at(step: int, total: int, warmup: int, peak: float, floor: float) -> float:
    """Linear warm-up to `peak`, then cosine decay to `floor`."""
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * t))


def param_groups(model: nn.Module, weight_decay: float):
    """No weight decay on biases, normalization weights, positional embedding and [CLS] token."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or name.endswith(("pos_embed", "cls_token")) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def evaluate_split(model, norm, x, y, batch_size, amp):
    model.eval()
    loss_sum, correct = 0.0, 0
    for xb, yb in D.iterate_batches(x, y, batch_size, shuffle=False):
        with torch.autocast(x.device.type, enabled=amp):
            logits = model(norm(xb))
        loss_sum += nn.functional.cross_entropy(logits.float(), yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    return loss_sum / len(y), correct / len(y)


def train(model_name: str, seed: int, cfg: dict, out_root: str, max_train=None, epochs=None):
    tc = cfg["train"]
    epochs = epochs or tc["epochs"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(tc["amp"]) and dev.type == "cuda"
    run_dir = os.path.join(out_root, f"{model_name}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, "done.json")):
        print(f"[skip] {run_dir} already finished"); return run_dir

    # ---- data: identical split and normalization for every model
    tx, ty, _, _ = D.load_cifar10(cfg["data"]["root"])
    tr_idx, va_idx = D.stratified_split(ty, cfg["data"]["val_per_class"], cfg["data"]["split_seed"])
    split_info = D.check_split(tr_idx, va_idx, ty, len(ty))   # fails loudly if train/val overlap
    if max_train:                                     # only for quick smoke tests
        tr_idx, va_idx = tr_idx[:max_train], va_idx[:max_train // 4]
    mean, std = D.channel_stats(tx[tr_idx])
    xtr, ytr = D.to_tensor(tx[tr_idx]).to(dev), torch.from_numpy(ty[tr_idx]).to(dev)
    xva, yva = D.to_tensor(tx[va_idx]).to(dev), torch.from_numpy(ty[va_idx]).to(dev)

    set_seed(seed)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    model = build_model(model_name, cfg["models"].get(model_name, {})).to(dev)
    norm = D.Normalizer(mean, std).to(dev)
    opt = torch.optim.AdamW(param_groups(model, tc["weight_decay"]), lr=tc["lr"], betas=tuple(tc["betas"]))
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)
    steps_per_epoch = math.ceil(len(ytr) / tc["batch_size"])
    total_steps, warmup = epochs * steps_per_epoch, tc["warmup_epochs"] * steps_per_epoch

    start_epoch, best_acc, best_epoch = 0, -1.0, -1
    last = os.path.join(run_dir, "last.pt")
    if os.path.exists(last):                          # resume after a Colab disconnect
        ck = torch.load(last, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        gen.set_state(ck["gen"].cpu() if dev.type == "cpu" else ck["gen"])
        torch.set_rng_state(ck["torch_rng"])
        start_epoch, best_acc, best_epoch = ck["epoch"] + 1, ck["best_acc"], ck["best_epoch"]
        print(f"[resume] from epoch {start_epoch}")

    meta = {"model": model_name, "seed": seed, "params": count_parameters(model),
            "mean": mean.tolist(), "std": std.tolist(), "epochs": epochs, "device": str(dev),
            "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
            "torch": torch.__version__, "split": split_info, "config": cfg}
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{model_name} seed={seed}  params={meta['params']:,}  device={dev}")

    log_path = os.path.join(run_dir, "log.csv")
    if start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])

    loss_fn = nn.CrossEntropyLoss(label_smoothing=tc["label_smoothing"])
    aug = tc["augmentation"]
    for epoch in range(start_epoch, epochs):
        model.train(); t0 = time.time()
        loss_sum, correct, seen = 0.0, 0, 0
        for i, (xb, yb) in enumerate(D.iterate_batches(xtr, ytr, tc["batch_size"], True, gen)):
            step = epoch * steps_per_epoch + i
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup, tc["lr"], tc["min_lr"])
            xb = D.augment_batch(xb, aug["random_crop_padding"], aug["horizontal_flip"], gen)
            with torch.autocast(dev.type, enabled=amp):
                logits = model(norm(xb))
                loss = loss_fn(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt); scaler.update()
            loss_sum += loss.item() * len(yb); correct += (logits.argmax(1) == yb).sum().item(); seen += len(yb)
        va_loss, va_acc = evaluate_split(model, norm, xva, yva, 1000, amp)
        secs = time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{opt.param_groups[0]['lr']:.6g}", f"{loss_sum/seen:.4f}",
                                    f"{correct/seen:.4f}", f"{va_loss:.4f}", f"{va_acc:.4f}", f"{secs:.1f}"])
        if va_acc > best_acc:                          # keep the best clean-validation checkpoint
            best_acc, best_epoch = va_acc, epoch + 1
            torch.save({"model": model.state_dict(), "mean": mean, "std": std, "epoch": best_epoch,
                        "val_acc": best_acc}, os.path.join(run_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "gen": gen.get_state(), "torch_rng": torch.get_rng_state(), "epoch": epoch,
                    "best_acc": best_acc, "best_epoch": best_epoch}, last)
        print(f"ep {epoch+1:3d}/{epochs} loss {loss_sum/seen:.3f} train {correct/seen:.3f} "
              f"val {va_acc:.4f} best {best_acc:.4f}@{best_epoch} ({secs:.0f}s)")

    with open(os.path.join(run_dir, "done.json"), "w") as f:
        json.dump({"best_val_acc": best_acc, "best_epoch": best_epoch}, f, indent=2)
    os.remove(last)                                   # keep the repo/Drive small
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["plain_cnn", "resnet18", "vit"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=None, help="override, for smoke tests only")
    ap.add_argument("--max-train", type=int, default=None, help="subset size, for smoke tests only")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    train(a.model, a.seed, cfg, a.out, a.max_train, a.epochs)


if __name__ == "__main__":
    main()
