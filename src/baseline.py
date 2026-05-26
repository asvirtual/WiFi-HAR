import torch
from torch.nn import Conv2d, MaxPool2d, ReLU, Softmax, Dropout, Sequential, Linear, Flatten
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor
from dataset import CFR
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

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
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    
    def forward(self,x) -> torch.Tensor:
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
            Conv2d(kernel_size=4, stride=2, in_channels=6, out_channels=9),
            ReLU()
        )

        self.relu = ReLU()
        self.apply(self._init_weights)

        

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, x) -> torch.Tensor:
        x1 = self.relu(self.block1(x))
        x2 = self.relu(self.convBlock1(x))
        x3 = self.convBlock2(x)
        print(x1.shape, x2.shape, x3.shape)
        y = torch.cat((x1, x2, x3), dim=1)
        return y



model = BaselineNet()
dataset = CFR(folder="../data/doppler_traces/S1", transform=ToTensor())
total_len = len(dataset)

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [total_len * 0.6, total_len * 0.2, total_len * 0.2],
    generator=torch.Generator().manual_seed(42)
)

batch_size = 64
num_workers = 0
pin_memory = torch.cuda.is_available()

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)

valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)



opt = Adam(model.parameters(), lr=1e-3, weight_decay = 0)
loss_fn = CrossEntropyLoss()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
epochs=10
best_val = np.inf
for epoch in range(epochs):
    model.train()
    print(f"Epoch: {epoch+1}")
    train_iterator = tqdm(train_dataloader)
    for batch_x, batch_y in train_iterator:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        y_pred = model(batch_x)

        loss = loss_fn(y_pred, batch_y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        train_iterator.set_description(f"Train loss: {loss.detach().cpu().numpy()}")

    model.eval()
    with torch.no_grad():
        predictions = []
        true = []
        val_iterator = tqdm(valid_dataloader)
        for batch_x, batch_y in val_iterator:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            predictions.append(y_pred)
            true.append(batch_y)
            val_iterator.set_description(f"Validation loss: {loss.detach().cpu().numpy()}")
        predictions = torch.cat(predictions, dim=0)
        true = torch.cat(true, dim=0)
        val_loss = loss_fn(predictions, true)
        val_acc = None # TODO
        print(f"loss: {val_loss}, accuracy: {val_acc}")

    if val_loss < best_val:
        print("Saved Model")
        torch.save(model.state_dict(), "resnet50.pt")
        best_val = val_loss

