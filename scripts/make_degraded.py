"""Generate the 13 fixed test sets once (owner: Afnan).

Usage:  python -m scripts.make_degraded
"""
import yaml

import json, os

import numpy as np

from src.data import check_split, load_cifar10, stratified_split
from src.degradations import generate_test_sets

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    train_x, train_y, test_x, test_y = load_cifar10(cfg["data"]["root"])
    tr, va = stratified_split(train_y, cfg["data"]["val_per_class"], cfg["data"]["split_seed"])
    info = check_split(tr, va, train_y, len(train_y))
    os.makedirs("results", exist_ok=True)
    np.savez_compressed("results/split_indices.npz", train_idx=tr, val_idx=va)   # anyone can re-check the split
    json.dump(info, open("results/split_check.json", "w"), indent=2)
    print("split OK:", info["train"], "train /", info["val"], "val /", len(test_y), "test, overlap = 0")
    generate_test_sets(test_x, test_y, cfg["data"]["degraded_dir"], cfg["data"]["degradation_seed"])
