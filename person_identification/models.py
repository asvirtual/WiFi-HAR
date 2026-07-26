import torch
from torch.nn import Conv2d, Dropout2d, MaxPool2d, ReLU,Dropout, Sequential, Linear, Flatten, CrossEntropyLoss
import numpy as np



class InceptionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = MaxPool2d(kernel_size=2, stride=2)
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
        return torch.cat((x1, x2, x3), dim=1)




class InceptionCNN(torch.nn.Module):
    def __init__(self, num_classes=3):  
        super().__init__()
        
        self.backbone = Sequential(
            InceptionModule(),
            Conv2d(kernel_size=1, stride=1, padding=0, in_channels=52, out_channels=16),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            Dropout(0.2),
            Linear(in_features=34000, out_features=256),
            ReLU(),
            Dropout(0.1)   
        )
        
        
        self.classifier = Linear(in_features=256, out_features=num_classes)
        
        
        self.projector = Sequential(
            Linear(in_features=256, out_features=128),
            ReLU(),
            Linear(in_features=128, out_features=64)
        )
        
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, return_projection=False):
        
        features = self.backbone(x)
        
        logits = self.classifier(features)
        
        
        if return_projection:
            projection = self.projector(features)
            return logits, projection
            
        return logits

