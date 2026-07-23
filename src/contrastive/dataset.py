import torch
from torch.utils.data import DataLoader, Dataset
from torch import long
import pickle
import numpy as np
import os, random
from collections import defaultdict

'''
Folder/file naming and structure
    Subfolders: S[n][abc] -> n is the split index among the whole dataset, a/b/c is the "campaign" id (same setting, different day)
    Files: s[n][abc]_[activity]_stream_[antenna] -> activity is the activity id (e.g. C = sitting down/standing up) and antenna 
        is the id of the antenna (0, 1, 2 or 3)

'''

SAMPLE_SIZE_ROWS = 340
SAMPLE_SIZE_COLS = 100

class SpectogramAugmentation:
    def __init__(self, tprob=0.5, fprob=0.5, max_tmask=40, max_fmask=5):
        self.tprob = tprob
        self.fprob = fprob
        self.max_tmask = max_tmask
        self.max_fmask = max_fmask

    def __call__(self, x):
        h,w = x.shape[1], x.shape[2] # height and width of the spectrogram
        # Time Masking:
        if random.random() < self.tprob:
            t_range = random.randint(5, self.max_tmask)
            t0 = random.randint(0, h - t_range) 
            x[:, t0:t0+t_range, :] = 0.0
        # Frequency Masking:
        if random.random() < self.fprob:
            f_range = random.randint(2, self.max_fmask)
            f0 = random.randint(0, w - f_range)
            x[:, :, f0:f0+f_range] = 0.0
        return x



class CFR(Dataset):
    # LABEL_MAP = {
    #     "C": 0,
    #     "E": 1,
    #     "H": 2,
    #     "J1": 3,
    #     "J2": 3,
    #     "L": 4,
    #     "R": 5,
    #     "S": 6,
    #     "W": 7,
    # }

    LABEL_MAP = {
        # "C": 0,
        "E": 0,
        # "H": 2,
        # "J1": 3,
        # "J2": 3,
        # "L": 4,
        "S": 1,
        "W": 2,
        "R": 3,
        "J": 4
    }

    # def sliding_window(self, matrix, window_size, stride):
    #     total_frames, pack = matrix.shape
    #     windows = []
    #     for index in range(0, total_frames - window_size + 1, stride):
    #         window = matrix[index:index + window_size, :]
    #         windows.append(window)
    #         #print(window.shape[0], window.shape[1])
    #     return torch.stack(windows)

    def __init__(self, folder, campaigns, split_mode="train", stride=5, transform=None, max_samples=None, use_multi_antenna=False):

        self.matrices = []
        self.window_info = []
        self.split_mode = split_mode
        self.use_multi_antenna = use_multi_antenna
        self.grouped_matrices = defaultdict(dict)
        self.grouped_window_info = []
        self.transform = transform
        self.max_samples = max_samples

        for campaign in campaigns:
            campaign_path = f"./{folder}{campaign}"
            if not os.path.exists(campaign_path):
                continue

            for file in os.listdir(campaign_path):
                with open(f"{campaign_path}/{file}", "rb") as f:
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

                    label = file.split("_")[1]
                    label = label[0].upper()

                    if label not in self.LABEL_MAP:
                        continue

                    label_id = torch.tensor(self.LABEL_MAP[label]).long()

                    if self.use_multi_antenna:
                        event_key, antenna_key = self._split_event_and_antenna(file)
                        self.grouped_matrices[event_key][antenna_key] = matrix
                    else:
                        mat_idx = len(self.matrices)
                        self.matrices.append(matrix)

                        total_frames = matrix.shape[0]
                        for index in range(0, total_frames - SAMPLE_SIZE_ROWS + 1, stride):
                            self.window_info.append((mat_idx, index, label_id))

        if self.use_multi_antenna:
            for event_key, antenna_matrices in self.grouped_matrices.items():
                if len(antenna_matrices) < 2:
                    continue

                ordered_antenna_keys = sorted(antenna_matrices.keys(), key=self._sort_antenna_key)
                min_length = min(matrix.shape[0] for matrix in antenna_matrices.values())
                if min_length < SAMPLE_SIZE_ROWS:
                    continue

                if event_key.split("_")[1] not in self.LABEL_MAP: continue

                label_id = torch.tensor(self.LABEL_MAP[event_key.split("_")[1]]).long()
                for index in range(0, min_length - SAMPLE_SIZE_ROWS + 1, stride):
                    self.grouped_window_info.append((event_key, index, label_id, ordered_antenna_keys))

    def _split_event_and_antenna(self, file_name):
        stem = os.path.splitext(file_name)[0]
        if "_" not in stem:
            return stem, "0"
        event_key, antenna_key = stem.rsplit("_", 1)
        return event_key, antenna_key

    def _sort_antenna_key(self, antenna_key):
        if antenna_key.isdigit():
            return (0, int(antenna_key))
        antenna_order = {"a": 0, "b": 1, "c": 2, "d": 3}
        return (1, antenna_order.get(antenna_key.lower(), antenna_key))

    def __len__(self):
        if self.use_multi_antenna:
            if self.max_samples is None:
                return len(self.grouped_window_info)
            return min(self.max_samples, len(self.grouped_window_info))
        if self.max_samples is None:
            return len(self.window_info)
        return min(self.max_samples, len(self.window_info))

    def __getitem__(self, idx):
        if self.use_multi_antenna:
            event_key, index, label_id, antenna_keys = self.grouped_window_info[idx]
            views = []
            for antenna_key in antenna_keys:
                matrix = self.grouped_matrices[event_key][antenna_key]
                x = matrix[index:index + SAMPLE_SIZE_ROWS, :].clone()
                x = x.unsqueeze(0)
                if self.transform:
                    x = self.transform(x)
                views.append(x)
            return torch.stack(views, dim=0), label_id

        mat_idx, index, label_id = self.window_info[idx]
        matrix = self.matrices[mat_idx]
        # shape of the input: Time x Frequency (340x100)
        x = matrix[index:index + SAMPLE_SIZE_ROWS, :].clone() # create a copy such that we don't modify the original matrix when applying transforms
        y = label_id

        x = x.unsqueeze(0)  # Add channel dimension, now we have shape: 1 x Time x Frequency (1x340x100)
        if self.transform:
            x = self.transform(x)
        return x, y 