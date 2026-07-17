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
        "C": 0,
        "E": 1,
        "H": 2,
        "J1": 3,
        "J2": 3,
        "L": 4,
        "R": 5,
        "S": 6,
        "W": 7,
    }

    def sliding_window(self, matrix, window_size, stride):
        total_frames, pack = matrix.shape
        windows = []
        for index in range(0, total_frames - window_size + 1, stride):
            window = matrix[index:index + window_size, :]
            windows.append(window)
        return torch.stack(windows)

    def __init__(self, folder, campaigns, split_mode="train", stride=50, transform=None, max_samples=None):

        x_list = []
        y_list = []

        for campaign in campaigns:
            if not os.path.exists(f"./{folder}{campaign}"):
                continue
            for file in os.listdir(f"./{folder}{campaign}"):
                with open(f"{folder}{campaign}/{file}", "rb") as f:
                    matrix = torch.from_numpy(pickle.load(f)).float()
                    if folder == "../data/doppler_traces/S1":
                        if split_mode == "val":
                            end_point = matrix.shape[0] // 2
                            matrix = matrix[:end_point, :]

                        if split_mode == "test":
                            start_point = matrix.shape[0] // 2
                            matrix = matrix[start_point:, :]

                    if matrix.shape[0] < SAMPLE_SIZE_ROWS:
                        continue  # Skip files that are too short

                    label_id = torch.tensor(self.LABEL_MAP[file.split("_")[1]]).long()
                    windows = self.sliding_window(matrix, SAMPLE_SIZE_ROWS, stride)
                    x_list.append(windows)
                    labels = torch.full((windows.shape[0],), label_id, dtype=torch.long)
                    y_list.append(labels)

                    
                #print(f"Loaded {file} -> extracted {windows.shape[0]} windows with label {label_id}")
        
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
        x = x.unsqueeze(0)  # Add channel dimension
        if self.transform:
            x = self.transform(x)
        return x, y 