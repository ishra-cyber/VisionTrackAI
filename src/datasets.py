"""PyTorch datasets built from the split CSVs written by scripts/prepare_data.py."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment import augment
from .preprocessing import apply_clahe, normalize, read_mask, read_rgb, resize


class SegDataset(Dataset):
    """Returns (image[3,H,W], target[2,H,W]) with target0 = disc (disc ∪ cup), target1 = cup."""

    def __init__(self, csv_path, size=512, train=False):
        self.df = pd.read_csv(csv_path)
        self.size, self.train = size, train

    def __len__(self):
        return len(self.df)

    def load(self, i):
        r = self.df.iloc[i]
        img = read_rgb(r["fundus"])
        disc = read_mask(r["disc"], img.shape)
        cup = read_mask(r["cup"], img.shape)
        disc = np.maximum(disc, cup)
        img = resize(apply_clahe(img), self.size)
        return img, resize(disc, self.size, True), resize(cup, self.size, True)

    def __getitem__(self, i):
        img, disc, cup = self.load(i)
        if self.train:
            img, (disc, cup) = augment(img, (disc, cup))
        x = torch.from_numpy(normalize(img))
        y = torch.from_numpy(np.stack([disc, cup]).astype(np.float32))
        return x, y


class ClsDataset(Dataset):
    """Returns (image[3,H,W], label float)."""

    def __init__(self, csv_path, size=384, train=False):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["label"].isin([0, 1])].reset_index(drop=True)
        self.size, self.train = size, train

    def __len__(self):
        return len(self.df)

    @property
    def labels(self):
        return self.df["label"].astype(int).values

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = read_rgb(r["fundus"])
        if img.shape[0] != self.size or img.shape[1] != self.size:
            img = resize(img, self.size)
        img = apply_clahe(img)
        if self.train:
            img, _ = augment(img)
        return torch.from_numpy(normalize(img)), torch.tensor(float(r["label"]))
