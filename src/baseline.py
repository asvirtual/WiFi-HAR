import torch
from torch.nn import Conv2d, MaxPool2d, ReLU, Softmax, Dropout, Sequential, Linear, Flatten
import numpy as np
import torch.utils.data 


class BaselineNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential=Sequential(
            InceptionModule(),
            Conv2d(kernel_size=1,stride=1,padding=0,out_channels=3,in_channels=15),
            ReLU(),
            Flatten(),
            Dropout(0.2),
            Linear(in_features=25500, out_features=128),
            ReLU(),
            Linear(in_features=128,out_features=8),
        )
        self.softmax = Softmax(dim=1)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    
    def forward(self,x):
        y=self.softmax(self.sequential(x))
        return y        


class InceptionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Max Pool
        self.block1 = MaxPool2d(kernel_size=2, stride=2)
        
        # Conv @5 (2x2) stride 2
        self.convBlock1 = Conv2d(kernel_size=2, stride=2, out_channels=5, in_channels=1)

        # Conv 3@ (1x1) stride 1 -> 6@ (2x2) stride 1 -> 9@ (4x4) stride 2
        self.convBlock2 = Sequential(                   
            Conv2d(kernel_size=1, stride=1, in_channels=1, out_channels=3),
            ReLU(),
            Conv2d(kernel_size=2, stride=1, in_channels=3, out_channels=6),
            ReLU(),
            Conv2d(kernel_size=4, stride=2, in_channels=6, out_channels=9, padding="same"),
            ReLU()
        )

        self.relu = ReLU()
        self.apply(self._init_weights)

        

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, x):
        x1 = self.relu(self.block1(x))
        x2 = self.relu(self.convBlock1(x))
        x3 = self.convBlock2(x)
        print(x1.shape, x2.shape, x3.shape)
        y = torch.cat((x1, x2, x3), dim=1)
        return y



model = BaselineNet()
