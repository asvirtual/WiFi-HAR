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
        "E": 6,
        "H": 7,
    }

    def sliding_window(self, matrix, window_size, step_size):
        total_frames, pack = matrix.shape
        windows = []
        for index in range(0, total_frames - window_size + 1, step_size):
            window = matrix[index:index + window_size, :]
            windows.append(window)
        return torch.stack(windows)

    def sliding_window(self, matrix, window_size, step_size):
        total_frames, pack = matrix.shape
        windows = []
        for index in range(0, total_frames - window_size + 1, step_size):
            window = matrix[index:index + window_size, :]
            windows.append(window)
        return torch.stack(windows)

    def __init__(self, folder, transform=None, max_samples=None):

        x_list = []
        y_list = []

        for campaign in ["a", "b", "c"]:
            for file in os.listdir(f"./{folder}{campaign}"):
                with open(f"{folder}{campaign}/{file}", "rb") as f:
                    matrix = torch.from_numpy(pickle.load(f)).float()
                    label_id = torch.tensor(self.LABEL_MAP[file.split("_")[1]]).long()
                    windows = self.sliding_window(matrix, SAMPLE_SIZE_ROWS, 50)

                    x_list.append(windows)
                    labels = torch.full((windows.shape[0],), label_id, dtype=torch.long)
                    y_list.append(labels)

                    
                print(f"Loaded {file} -> extracted {windows.shape[0]} windows with label {label_id}")
        
        self.x = torch.cat(x_list, dim=0)
        self.y = torch.cat(y_list, dim=0)

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