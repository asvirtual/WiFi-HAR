import torch
import matplotlib.pyplot as plt
from torch.nn import AdaptiveAvgPool2d, Conv2d, Dropout2d, MaxPool2d, ReLU, Softmax, Dropout, Sequential, Linear, Flatten, BatchNorm2d, BatchNorm1d
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor
from dataset import CFR_PersonID_4Channels as CFR_PersonID
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

import matplotlib.pyplot as plt

from torch.nn import Conv2d, MaxPool2d, ReLU, Dropout, Sequential, Linear, Flatten, InstanceNorm1d,InstanceNorm2d


class InceptionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = MaxPool2d(kernel_size=2, stride=2)
        
        # ORA in_channels = 4 (Le 4 Antenne!)
        self.convBlock1 = Conv2d(kernel_size=2, stride=2, in_channels=4, out_channels=16)

        self.convBlock2 = Sequential(                   
            Conv2d(kernel_size=1, stride=1, in_channels=4, out_channels=8),
            ReLU(),
            Conv2d(kernel_size=2, stride=1, in_channels=8, out_channels=16, padding='same'),
            ReLU(),
            Conv2d(kernel_size=4, stride=2, in_channels=16, out_channels=32, padding=1),
            ReLU(),
        )
        self.relu = ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.relu(self.convBlock1(x))
        x3 = self.convBlock2(x)
        # Output shape: 4 (x1) + 16 (x2) + 32 (x3) = 52 canali
        return torch.cat((x1, x2, x3), dim=1)


import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Dropout, Sequential, Linear, Flatten, BatchNorm1d, InstanceNorm2d

# ... (Qui sopra rimane il tuo InceptionModule con in_channels=4 esattamente come lo hai già) ...

import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Dropout, Sequential, Linear, Flatten, BatchNorm1d, BatchNorm2d, InstanceNorm1d, InstanceNorm2d

# ... (Qui sopra rimane intatto il tuo InceptionModule) ...

class BaselineNet(torch.nn.Module):
    def __init__(self, num_classes=3):  # 3 Uscite per P0, P1, P2
        super().__init__()
        
        self.backbone = Sequential(
            InceptionModule(),
            Conv2d(kernel_size=1, stride=1, padding=0, in_channels=52, out_channels=16),
            #BatchNorm2d(16),
            #InstanceNorm2d(16),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            Dropout(0.1),  # Il tuo Dropout robusto allo 0.4!
            Linear(in_features=34000, out_features=256),
            #BatchNorm1d(256),
            #InstanceNorm1d(256),
            ReLU(),
            Dropout(0.2)   # Il tuo Dropout robusto allo 0.4!
        )
        
        
        self.classifier = Linear(in_features=256, out_features=num_classes)
        
        
        self.projector = Sequential(
            Linear(in_features=256, out_features=128),
            ReLU(),
            Linear(in_features=128, out_features=64)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, return_projection=False):
        
        features = self.backbone(x)
        
        logits = self.classifier(features)
        
        
        if return_projection:
            projection = self.projector(features)
            return logits, projection
            
        return logits