import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import os

# Define the dimensions of a single input window (spectrogram crop)
# SAMPLE_SIZE_ROWS: Time frames (temporal dimension)
# SAMPLE_SIZE_COLS: Frequency bins (Doppler shifts)
SAMPLE_SIZE_ROWS = 340
SAMPLE_SIZE_COLS = 100

class CFR_PersonID_4Channels(Dataset):
    """
    Custom PyTorch Dataset for Wi-Fi Channel Frequency Response (CFR) data.
    It loads micro-Doppler spectrograms from 4 spatial antennas (channels),
    performs static clutter removal, and applies sliding window segmentation.
    """
    
    # Mapping subjects from data campaigns to target classification classes (Person IDs)
    # This groups different recording sessions into specific human identities.
    PERSON_MAP = {
        "S1": 0, "S2": 0, "S4": 0, "S6": 0,  
        "S3": 1, "S5": 1,                    
        "S7": 2                              
    }

    def sliding_window(self, matrix, window_size, stride):
        """
        Extracts overlapping temporal windows from the continuous CFR stream.
        Input matrix shape expected: [channels, total_time_frames, frequency_bins]
        Returns a stacked tensor of shape: [num_windows, channels, window_size, frequency_bins]
        """
        _, total_frames, _ = matrix.shape
        windows = []
        for index in range(0, total_frames - window_size + 1, stride):
            window = matrix[:, index:index + window_size, :]
            windows.append(window)
        return torch.stack(windows) if len(windows) > 0 else torch.empty(0)

    def __init__(self, folder="../data/doppler_traces/", campaigns=["S1a", "S3a", "S7a"], 
                 target_activities=["W", "R"], split_mode="train", stride=50):
        
        x_list = []
        y_list = []
        person_counts = {0: 0, 1: 0, 2: 0}

        for campaign in campaigns:
            campaign_path = os.path.join(folder, campaign)
            if not os.path.exists(campaign_path): continue
            
            # Extract subject ID (e.g., 'S1') and map it to the corresponding Person class (0, 1, or 2)
            subject_code = campaign[:2]
            if subject_code not in self.PERSON_MAP: continue
            person_id = self.PERSON_MAP[subject_code]

            # Parse files and group the 4 antenna streams by activity
            activity_files = {}
            for file in os.listdir(campaign_path):
                if not file.endswith(".txt"): continue
                parts = file.split("_")
                act = parts[1]
                stream_idx = parts[3].split(".")[0] 
                
                if act not in activity_files: activity_files[act] = {}
                activity_files[act][stream_idx] = file

            # Process each activity that has a complete set of 4 antennas
            for act, streams in activity_files.items():
                if target_activities and act not in target_activities: continue
                if len(streams) < 4: continue # Skip if any of the 4 RX antennas is missing

                # Load the micro-Doppler traces for all 4 antennas
                matrices = []
                for i in range(4):
                    filepath = os.path.join(campaign_path, streams[str(i)])
                    with open(filepath, "rb") as f:
                        matrices.append(torch.from_numpy(pickle.load(f)).float())
                
                # Stack the 4 matrices into a single multi-channel tensor
                # Shape becomes: [4, total_frames, 100]
                multi_channel_matrix = torch.stack(matrices, dim=0)
                total_frames = multi_channel_matrix.shape[1]

                # -------------------------------------------------------------
                # TEMPORAL SPLIT LOGIC (Strict Data Leakage Prevention)
                # Slices the continuous time-series stream chronologically
                # Train: 0%-70% | Val: 70%-85% | Test: 85%-100%
                # -------------------------------------------------------------
                if split_mode == "train":
                    multi_channel_matrix = multi_channel_matrix[:, :int(total_frames * 0.70), :]
                elif split_mode == "val":
                    multi_channel_matrix = multi_channel_matrix[:, int(total_frames * 0.70):int(total_frames * 0.85), :]
                elif split_mode == "test":
                    multi_channel_matrix = multi_channel_matrix[:, int(total_frames * 0.85):, :]

                # Skip if the remaining sequence is too short to form even a single window
                if multi_channel_matrix.shape[1] < SAMPLE_SIZE_ROWS: continue

                # -------------------------------------------------------------
                # Computes the temporal mean across the sequence and subtracts it.
                # This removes the static background reflections (walls, furniture) 
                # leaving only the dynamic kinetic features (human motion).
                # -------------------------------------------------------------
                mean_profile = multi_channel_matrix.mean(dim=1, keepdim=True)
                multi_channel_matrix = multi_channel_matrix - mean_profile
               
                # Generate temporal overlapping windows
                windows = self.sliding_window(multi_channel_matrix, SAMPLE_SIZE_ROWS, stride)
                if windows.shape[0] == 0: continue

                # Append features and correctly mapped labels
                x_list.append(windows)
                labels = torch.full((windows.shape[0],), person_id, dtype=torch.long)
                y_list.append(labels)
                person_counts[person_id] += windows.shape[0]
                
        # Concatenate all accumulated batches into final dataset tensors
        self.x = torch.cat(x_list, dim=0)
        self.y = torch.cat(y_list, dim=0)
        
        # Print summary statistics to verify class balance and dataset size
        print(f"Dataset [{split_mode.upper()}] - Windows: {len(self.x)} | P0: {person_counts[0]}, P1: {person_counts[1]}, P2: {person_counts[2]}")

    def __len__(self): 
        """Returns the total number of samples (windows) in the dataset."""
        return len(self.x)

    def __getitem__(self, idx):
        """Returns a single tuple (spectrogram_window, label) corresponding to the index."""
        return self.x[idx], self.y[idx]