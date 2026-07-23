import torch
from torch.utils.data import DataLoader, Dataset
from torch import long
import pickle
import numpy as np
import os, random

'''
Folder/file naming and structure
    Subfolders: S[n][abc] -> n is the split index among the whole dataset, a/b/c is the "campaign" id (same setting, different day)
    Files: s[n][abc]_[activity]_stream_[antenna] -> activity is the activity id (e.g. C = sitting down/standing up) and antenna 
        is the id of the antenna (0, 1, 2 or 3)

'''

SAMPLE_SIZE_ROWS = 340
SAMPLE_SIZE_COLS = 100

class SpectogramAugmentation2:
    def __init__(self, tprob=0.6, fprob=0.6, max_tmask=15, max_fmask=8,
                 noise_prob=0.4, gain_prob=0.4, noise_std=0.01,
                 gain_range=(0.9, 1.1)):
        self.tprob = tprob
        self.fprob = fprob
        self.max_tmask = max_tmask
        self.max_fmask = max_fmask
        self.noise_prob = noise_prob
        self.gain_prob = gain_prob
        self.noise_std = noise_std
        self.gain_range = gain_range

    def __call__(self, x):
        x = x.clone()
        h, w = x.shape[1], x.shape[2]  # height and width of the spectrogram

        if random.random() < self.gain_prob:
            scale = random.uniform(*self.gain_range)
            x = x * scale

        if random.random() < self.noise_prob:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        if random.random() < self.tprob:
            t_range = random.randint(5, self.max_tmask)
            t0 = random.randint(0, h - t_range)
            x[:, t0:t0 + t_range, :] = 0.0

        if random.random() < self.fprob:
            f_range = random.randint(2, self.max_fmask)
            f0 = random.randint(0, w - f_range)
            x[:, :, f0:f0 + f_range] = 0.0

        return x


class SpectogramAugmentation:
    def __init__(self, tprob=0.5, fprob=0.5, max_tmask=20, max_fmask=10):
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

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Normalizza: (x - mean) / std
        return (tensor - self.mean) / self.std



class CFR(Dataset):
    LABEL_MAP = {
        #"C": 7,
        "E": 0,
        #"H": 5,
        "S": 1,
        "W": 2,
        "R": 3, 
        "J": 4,
        #"L":6
    }

    # def sliding_window(self, matrix, window_size, stride):
    #     total_frames, pack = matrix.shape
    #     windows = []
    #     for index in range(0, total_frames - window_size + 1, stride):
    #         window = matrix[index:index + window_size, :]
    #         windows.append(window)
    #         #print(window.shape[0], window.shape[1])
    #     return torch.stack(windows)

    def __init__(self, folder, campaigns, split_mode="train", stride=5, transform=None, max_samples=None):

        self.matrix_groups = []
        self.window_info = []
        self.split_mode = split_mode

        for campaign in campaigns:
            if not os.path.exists(f"./{folder}{campaign}"):
                continue
            groups = {}
            for file in os.listdir(f"./{folder}{campaign}"):
                # split the matrices into groups based on the room/action and then on the antenna index
                prefix, stream_channel = file.rsplit("_stream_", 1)
                antenna_idx = int(stream_channel.split(".")[0])
                if prefix not in groups:
                    groups[prefix] = {}
                groups[prefix][antenna_idx] = file

            for prefix, ant_dict in groups.items():
                if len(ant_dict) < 4:
                    continue

                matrices = []
                skip_group = False
                for antenna_idx in range(4):
                    file = ant_dict[antenna_idx]
                    with open(f"{folder}{campaign}/{file}", "rb") as f:
                        matrix = torch.from_numpy(pickle.load(f)).float()
                        if split_mode == "val":
                            end_point = matrix.shape[0] // 2
                            matrix = matrix[:end_point, :]

                        if split_mode == "test":
                            start_point = matrix.shape[0] // 2
                            matrix = matrix[start_point:, :]

                        if matrix.shape[0] < SAMPLE_SIZE_ROWS:
                            skip_group=True
                            break  # Skip files that are too short

                        matrices.append(matrix)

                if skip_group or len(matrices) < 4:
                    continue

                label = prefix.split("_")[1]
                label = label[0].upper()

                if label not in self.LABEL_MAP:
                    continue

                label_id = torch.tensor(self.LABEL_MAP[label]).long()

                group_idx = len(self.matrix_groups)
                self.matrix_groups.append(matrices)

                min_len = min(m.shape[0] for m in matrices)
                for index in range(0, min_len - SAMPLE_SIZE_ROWS + 1, stride):
                    self.window_info.append((group_idx, index, label_id))

        self.transform = transform
        self.max_samples = max_samples

    def __len__(self):
        if self.max_samples is None:
            return len(self.window_info)
        return min(self.max_samples, len(self.window_info))

    def __getitem__(self, idx):
        group_idx, index, label_id = self.window_info[idx]
        matrices = self.matrix_groups[group_idx]

        x_list = []
        for matrix in matrices:
            window = matrix[index:index + SAMPLE_SIZE_ROWS, :].clone()  # create a copy such that we don't modify the original matrix when applying transforms
            window = window.unsqueeze(0)  # Add channel dimension, now we have shape: 1 x Time x Frequency (1x340x100)
            if self.transform:
                window = self.transform(window)
            x_list.append(window)
        # shape of the input: Time x Frequency (340x100)
        x = torch.stack(x_list, dim=0)  # shape: 4 x Time x Frequency (4x340x100)
        y = label_id
        return x, y 