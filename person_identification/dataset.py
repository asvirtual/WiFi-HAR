import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import os

SAMPLE_SIZE_ROWS = 340
SAMPLE_SIZE_COLS = 100

class CFR_PersonID_4Channels(Dataset):
    PERSON_MAP = {
        "S1": 0, "S2": 0, "S4": 0, "S6": 0,  # Persona 0
        "S3": 1, "S5": 1,                     # Persona 1
        "S7": 2                              # Persona 2
    }

    def sliding_window(self, matrix, window_size, stride):
        # Ora la matrice è [4_canali, tempo, frequenza]
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
            
            subject_code = campaign[:2]
            if subject_code not in self.PERSON_MAP: continue
            person_id = self.PERSON_MAP[subject_code]

            # Raggruppiamo i file per attività per unire le 4 antenne
            activity_files = {}
            for file in os.listdir(campaign_path):
                if not file.endswith(".txt"): continue
                parts = file.split("_")
                act = parts[1]
                stream_idx = parts[3].split(".")[0] # Prende '0', '1', '2', '3'
                
                if act not in activity_files: activity_files[act] = {}
                activity_files[act][stream_idx] = file

            for act, streams in activity_files.items():
                if target_activities and act not in target_activities: continue
                if len(streams) < 4: continue # Salta se mancano antenne

                # Carichiamo i 4 stream e li impiliamo
                matrices = []
                for i in range(4):
                    filepath = os.path.join(campaign_path, streams[str(i)])
                    with open(filepath, "rb") as f:
                        matrices.append(torch.from_numpy(pickle.load(f)).float())
                
                # Creiamo il tensore a 4 canali: [4, time_frames, 100_freq]
                multi_channel_matrix = torch.stack(matrices, dim=0)
                total_frames = multi_channel_matrix.shape[1]

                # Taglio Temporale senza Leakage
                if split_mode == "train":
                    multi_channel_matrix = multi_channel_matrix[:, :int(total_frames * 0.70), :]
                elif split_mode == "val":
                    multi_channel_matrix = multi_channel_matrix[:, int(total_frames * 0.70):int(total_frames * 0.85), :]
                elif split_mode == "test":
                    multi_channel_matrix = multi_channel_matrix[:, int(total_frames * 0.85):, :]

                if multi_channel_matrix.shape[1] < SAMPLE_SIZE_ROWS: continue

                # Calcoliamo la media lungo l'asse del tempo (dim=1)
                # Il risultato è un profilo [4, 1, 100] (l'impronta della stanza)
                mean_profile = multi_channel_matrix.mean(dim=1, keepdim=True)
                
                # Sottraiamo l'impronta della stanza dal segnale
                multi_channel_matrix = multi_channel_matrix - mean_profile
                # =========================================================

                # =========================================================
                
                windows = self.sliding_window(multi_channel_matrix, SAMPLE_SIZE_ROWS, stride)
                if windows.shape[0] == 0: continue

                x_list.append(windows)
                labels = torch.full((windows.shape[0],), person_id, dtype=torch.long)
                y_list.append(labels)
                person_counts[person_id] += windows.shape[0]
        self.x = torch.cat(x_list, dim=0)
        self.y = torch.cat(y_list, dim=0)
        
        print(f"Dataset [{split_mode.upper()}] - Finestre: {len(self.x)} | P0: {person_counts[0]}, P1: {person_counts[1]}, P2: {person_counts[2]}")

    def __len__(self): return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]