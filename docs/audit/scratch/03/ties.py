"""Read-only: how tied are the brightest MNIST/PneumoniaMNIST pixels used by size_matched (k = 9)?"""
import numpy as np
import torch
from xai.data import shortcut_split

for base in ["mnist", "pneumonia_mnist"]:
    ds = shortcut_split(base, "patch", "test")
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:100].tolist()
    tied, rows, digit_frac = [], [], []
    for i in idx:
        v = ds.object_map(i).reshape(-1)
        top = np.sort(v)[::-1]
        tied.append(int((v >= top[8] - 1e-9).sum()))  # pixels tied with or above the 9th brightest
        order = np.argsort(-v, kind="stable")[:9]
        rows.append(np.mean(order // 28))
        digit_frac.append(float((v > 0.1).mean()))
    print(base, "pixels >= 9th-brightest value: median", np.median(tied), "max", max(tied),
          "| mean row of chosen 9:", round(float(np.mean(rows)), 1),
          "| object (>0.1) share of image:", round(float(np.mean(digit_frac)), 3))
