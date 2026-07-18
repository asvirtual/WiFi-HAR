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
            #print(window.shape[0], window.shape[1])
        return torch.stack(windows)

    def __init__(self, folder, campaigns, split_mode="train", stride=5, transform=None, max_samples=None):

        self.matrices = []
        self.window_info = []

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
                    mat_idx = len(self.matrices)
                    self.matrices.append(matrix)

                    total_frames = matrix.shape[0]
                    for index in range(0, total_frames - SAMPLE_SIZE_ROWS + 1, stride):
                        self.window_info.append((mat_idx, index, label_id))

        self.transform = transform
        self.max_samples = max_samples

    def __len__(self):
        if self.max_samples is None:
            return len(self.window_info)
        return min(self.max_samples, len(self.window_info))

    def __getitem__(self, idx):
        mat_idx, index, label_id = self.window_info[idx]
        matrix = self.matrices[mat_idx]
        x = matrix[index:index + SAMPLE_SIZE_ROWS, :]
        y = label_id
        x = x.unsqueeze(0)  # Add channel dimension
        if self.transform:
            x = self.transform(x)
        return x, y 