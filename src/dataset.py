import torch
from torch.utils.data import DataLoader, Dataset
from torch import long
import pickle
import numpy as np
import os

'''
Folder/file naming and structure
    Subfolders: S[n][abc] -> n is the split index among the whole dataset, a/b/c is the "campaign" id (same setting, different day)
    Files: s[n][abc]_[activity]_stream_[antenna] -> activity is the activity id (e.g. C = sitting down/standing up) and antenna 
        is the id of the antenna (0, 1, 2 or 3)

'''


class CFR(Dataset):
    LABEL_MAP = {
        "W": 0,
        "R": 1,
        "J1": 2,
        "J2": 2,
        "L": 3,
        "S": 4,
        "C": 5,
        "G": 6,
        "E": 7
    }

    def __init__(self, folder, transform=None, max_samples=None):
        dim = 0
        for campaign in ["a", "b", "c"]:
            dim += len(os.listdir(f"{folder}{campaign}")) 

        self.x = torch.zeros((dim, 340, 100))
        self.y = torch.zeros(dim).long()

        for ci, campaign in enumerate(["a", "b", "c"]):
            for idx, file in enumerate(os.listdir(f"./{folder}{campaign}")):
                print(ci, idx)
                with open(f"{folder}{campaign}/{file}", "rb") as f:
                    self.x[idx * (ci + 1)] = torch.from_numpy(pickle.load(f)).float()
                    self.y[idx * (ci + 1)] = torch.tensor(self.LABEL_MAP[file.split("_")[1].split("_")[0]]).long()

        self.transform = transform
        self.max_samples = max_samples

    def __len__(self):
        if self.max_samples is None:
            return len(self.x)
        return min(self.max_samples, len(self.x))

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y 