import json
import torch
import matplotlib.pyplot as plt
from torch.nn import Conv2d, MaxPool2d, ReLU, Dropout, Sequential, Linear, Flatten, CrossEntropyLoss
import numpy as np
from torch.utils.data import DataLoader
from dataset import CFR
from torch.optim import Adam
from tqdm import tqdm

class BaselineNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential=Sequential(
            InceptionModule(),
            Conv2d(kernel_size=1,stride=1,padding=0,out_channels=3,in_channels=15),
            ReLU(),
            #MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            Dropout(0.2),
            Linear(in_features=25500, out_features=128),
            ReLU(),
            #Dropout(0.2),
            Linear(in_features=128,out_features=8),
        )
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    
    def forward(self,x) -> torch.Tensor:
        return self.sequential(x)


class InceptionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Max Pool
        self.block1 = MaxPool2d(kernel_size=2, stride=2)
        
        # Conv @5 (2x2) stride 2
        self.convBlock1 = Conv2d(kernel_size=2, stride=2, out_channels=5, in_channels=1)
        #self.bn1 = BatchNorm2d(num_features=5)

        # Conv 3@ (1x1) stride 1 -> 6@ (2x2) stride 1 -> 9@ (4x4) stride 2
        self.convBlock2 = Sequential(                   
            Conv2d(kernel_size=1, stride=1, in_channels=1, out_channels=3),
            ReLU(),
            Conv2d(kernel_size=2, stride=1, in_channels=3, out_channels=6, padding='same'),
            ReLU(),
            Conv2d(kernel_size=4, stride=2, in_channels=6, out_channels=9, padding=1),
            ReLU(),
            #Dropout2d(0.1)
        )

        self.relu = ReLU()
        self.apply(self._init_weights)

        

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, x) -> torch.Tensor:
        x1 = self.block1(x)
        x2 = self.relu(self.convBlock1(x))
        x3 = self.convBlock2(x)
        print(x1.shape, x2.shape, x3.shape)
        y = torch.cat((x1, x2, x3), dim=1)
        return y

if __name__ == "__main__":
    model = BaselineNet()

    train_dataset = CFR(folder="../data/doppler_traces/S1", campaigns=["a", "b"], split_mode="train")
    val_dataset = CFR(folder="../data/doppler_traces/S1", campaigns=["c"], split_mode="val")

    batch_size = 64
    num_workers = 0
    pin_memory = torch.cuda.is_available()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory)

    valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)


    opt = Adam(model.parameters(), lr=1e-3, weight_decay = 0)
    loss_fn = CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    epochs = 70
    patience = 7
    counter = 0

    best_val = np.inf
    checkpoint_path = "./models/baseline_model.pt"

    history = {
        "train": [],
        "val": [],
        "acc": []
    }


    for epoch in range(epochs):
        # TRAINING
        model.train()
        print(f"Epoch: {epoch+1}")

        cumtrain_loss = 0
        ntrain = 0
        train_iterator = tqdm(train_dataloader)
        for batch_x, batch_y in train_iterator:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            cumtrain_loss += loss.item() * batch_x.size(0)
            ntrain += batch_x.size(0)
            train_iterator.set_description(f"Train loss: {loss.item():.5f}")

        history["train"].append(cumtrain_loss / ntrain)

        # VALIDATION
        model.eval()
        cumval_loss = 0
        nval_correct = 0
        nval = 0

        with torch.no_grad():
            val_iterator = tqdm(valid_dataloader)
            for batch_x, batch_y in val_iterator:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                y_pred = model(batch_x)
                batch_loss = loss_fn(y_pred, batch_y)

                cumval_loss += batch_loss.item() * batch_x.size(0)
                nval += batch_x.size(0)

                predictions = y_pred.argmax(dim=1)
                nval_correct += (predictions == batch_y).sum().item()

                val_iterator.set_description(f"Validation loss: {batch_loss.item():.5f}")

            val_loss = cumval_loss / nval
            val_acc = nval_correct / nval
            history["val"].append(val_loss)
            history["acc"].append(val_acc)
            print(f"Validation loss: {val_loss}, accuracy: {val_acc}")

        if val_loss < best_val:
            print("Saved Model")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': val_loss
            }
            torch.save(checkpoint, checkpoint_path)
            best_val = val_loss
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print(f"[EARLY STOPPING] Validation loss hasn't improved for {patience} epochs.")
            break


    history_path = "plot_data/training_history_baseline.json"

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)