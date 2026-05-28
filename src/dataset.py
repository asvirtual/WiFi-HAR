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

SAMPLE_SIZE_ROWS = 340
SAMPLE_SIZE_COLS = 100
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
        "H": 7,
        "E": 8
    }

    def __init__(self, folder, transform=None, max_samples=None):
        dim = 1
        for campaign in ["a", "b", "c"]:
            for file in os.listdir(f"./{folder}{campaign}"):
                with open(f"{folder}{campaign}/{file}", "rb") as f:
                    data = torch.from_numpy(pickle.load(f)).float()
                    # for i in range((data.shape[0] // SAMPLE_SIZE_ROWS)):
                    #     dim += 1
                    dim += (data.shape[0] // SAMPLE_SIZE_ROWS)

        self.x = torch.zeros((dim, SAMPLE_SIZE_ROWS, SAMPLE_SIZE_COLS))
        self.y = torch.zeros(dim).long()

        # ci = 0, idx = 0, i = 0 -> index = 0 (first block first file first subfolder)
        # ci = 0, idx = 0, i = 1 -> index = 1 (second block first file first subfolder)
        # ...
        # ci = 0, idx = 1, i = 0 -> index =  (first block second file first subfolder)

        counter = 0
        for campaign in ["a", "b", "c"]:
            for file in os.listdir(f"./{folder}{campaign}"):
                with open(f"{folder}{campaign}/{file}", "rb") as f:
                    data = torch.from_numpy(pickle.load(f)).float()
                    label = self.LABEL_MAP[file.split("_")[1].split("_")[0]]
                    for i in range((data.shape[0] // SAMPLE_SIZE_ROWS)):
                        counter += 1
                        self.x[counter] = data[SAMPLE_SIZE_ROWS * i:SAMPLE_SIZE_ROWS * (i + 1),:]
                        self.y[counter] = torch.tensor(label).long()

        print(counter, dim)
        print(self.x[-1,:])
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