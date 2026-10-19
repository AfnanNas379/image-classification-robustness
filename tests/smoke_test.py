"""End-to-end smoke test on synthetic data (no download, CPU, ~2 min).

Checks that degradation, training, evaluation, metrics and plots all run
together. It does NOT produce real results. Usage: python -m tests.smoke_test
"""
import os, shutil, tempfile
import numpy as np, yaml
from src import data as D
from src.models import build_model, count_parameters, MODEL_NAMES


def fake_cifar(n_train=600, n_test=200, seed=0):
    rng = np.random.default_rng(seed)
    def make(n):
        y = rng.integers(0, 10, n)
        base = (np.stack([y * 25, 250 - y * 25, (y * 70) % 255], 1)[:, None, None, :]).astype(float)
        x = np.clip(base + rng.normal(0, 40, (n, 32, 32, 3)), 0, 255).astype(np.uint8)
        return x, y.astype(np.int64)
    (a, b), (c, d) = make(n_train), make(n_test)
    return a, b, c, d


def main():
    tmp = tempfile.mkdtemp()
    cfg = yaml.safe_load(open("configs/default.yaml"))
    cfg["data"]["degraded_dir"] = os.path.join(tmp, "degraded")
    cfg["data"]["val_per_class"] = 10
    cfg["train"]["warmup_epochs"] = 1
    fake = fake_cifar()
    D.load_cifar10 = lambda root="data": fake
    import src.train as T; T.D.load_cifar10 = D.load_cifar10

    for m in MODEL_NAMES:
        print(m, f"{count_parameters(build_model(m, cfg['models'].get(m, {}))):,} params")
    from src.degradations import generate_test_sets
    generate_test_sets(fake[2], fake[3], cfg["data"]["degraded_dir"], cfg["data"]["degradation_seed"])
    for m in MODEL_NAMES:
        for s in (0, 1):
            T.train(m, s, cfg, os.path.join(tmp, "runs"), epochs=2)
    cfg_path = os.path.join(tmp, "cfg.yaml"); yaml.safe_dump(cfg, open(cfg_path, "w"))
    import sys
    from src import evaluate, plots
    sys.argv = ["x", "--runs", os.path.join(tmp, "runs"), "--out", os.path.join(tmp, "results"), "--config", cfg_path]
    evaluate.main()
    sys.argv = ["x", "--results", os.path.join(tmp, "results"), "--runs", os.path.join(tmp, "runs"),
                "--degraded", cfg["data"]["degraded_dir"], "--out", os.path.join(tmp, "figures")]
    plots.main()
    print("SMOKE TEST PASSED; outputs in", tmp)


if __name__ == "__main__":
    main()
