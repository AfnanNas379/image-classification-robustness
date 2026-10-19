"""Checks that training resumed after an interruption gives identical weights. Usage: python -m tests.resume_check"""
import os, tempfile, torch, yaml, numpy as np
from tests.smoke_test import fake_cifar
from src import data as D
import src.train as T
fake = fake_cifar(); T.D.load_cifar10 = lambda root="data": fake
cfg = yaml.safe_load(open("configs/default.yaml")); cfg["data"]["val_per_class"]=10; cfg["train"]["warmup_epochs"]=1
a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
T.train("plain_cnn", 0, cfg, a, epochs=3)                     # uninterrupted
orig = T.evaluate_split; calls = {"n": 0}
def crash(*x, **k):
    calls["n"] += 1
    if calls["n"] == 2: raise RuntimeError("simulated Colab disconnect")
    return orig(*x, **k)
T.evaluate_split = crash
try: T.train("plain_cnn", 0, cfg, b, epochs=3)
except RuntimeError as e: print("interrupted:", e)
T.evaluate_split = orig
T.train("plain_cnn", 0, cfg, b, epochs=3)                     # resumes
la = open(os.path.join(a,"plain_cnn_seed0","log.csv")).read().splitlines()
lb = open(os.path.join(b,"plain_cnn_seed0","log.csv")).read().splitlines()
print("uninterrupted:", la[-1]); print("resumed:      ", lb[-1])
wa = torch.load(os.path.join(a,"plain_cnn_seed0","best.pt"),weights_only=False)["model"]
wb = torch.load(os.path.join(b,"plain_cnn_seed0","best.pt"),weights_only=False)["model"]
print("identical weights:", all(torch.equal(wa[k], wb[k]) for k in wa))
