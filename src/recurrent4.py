import json
import torch
import matplotlib.pyplot as plt
from torch.nn import LSTM, BatchNorm1d, BatchNorm2d, InstanceNorm2d, Conv2d, MaxPool2d, ReLU, Dropout, Sequential, Linear, Flatten, CrossEntropyLoss
from torchvision.transforms import Compose
import numpy as np
from torch.utils.data import DataLoader
from dataset2 import CFR, SpectogramAugmentation, Normalize
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from tqdm import tqdm

class ConvolutionalRecurrentNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = Sequential(
            InceptionModule(),
            Conv2d(kernel_size=4,stride=2,padding=0,out_channels=32,in_channels=15),  # 32, 84, 24
            BatchNorm2d(num_features=32), 
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2), # 32, 42, 12
        )
        # we will treat the 42 time steps as a sequence, and each time step has 32*12 features (number of filter * number of frequency bins)
        self.lstm = LSTM(input_size=32*12, hidden_size=64, num_layers=2, dropout=0.2, batch_first=True, bidirectional=True) 
        # for each of the temporal steps, we output a vector of size 128
        self.classificator=Sequential(
            Dropout(0.2),
            Linear(in_features=384, out_features=128), # head projection that maps the 128 features from the LSTM (bidirectional) to 128 features that merge those informations
            ReLU(),
            BatchNorm1d(num_features=128, momentum=0.01),
            Dropout(0.2),
            Linear(in_features=128,out_features=8),
        )
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    
    def forward(self,x) -> torch.Tensor:
        # convolutional network
        x = self.cnn(x)
        # we reshape the output of the CNN to be suitable for the LSTM: (batch_size, time_steps, channels * features)
        x = x.permute(0, 2, 1, 3)
        batch_size, time_steps, channels, features = x.size()
        x = x.reshape(batch_size, time_steps, channels * features)
        # recurrent layer
        x  = self.lstm(x)[0] # we only take the output of the last layer of the LSTM

        mean = torch.mean(x, dim=1) # we average the output of the LSTM over the time dimensiom
        max = torch.max(x, dim=1).values # we take the max over the time dimension
        std = torch.std(x, dim=1, unbiased=False) # we take the std over the time dimension

        x = torch.cat((mean, max, std), dim=-1) # we concatenate the mean, max and std to get a vector of size 128*3=384
        return self.classificator(x)


class InceptionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Max Pool
        self.block1 = MaxPool2d(kernel_size=2, stride=2)
        
        # Conv @5 (2x2) stride 2
        self.convBlock1 = Sequential(
            Conv2d(kernel_size=2, stride=2, out_channels=5, in_channels=1),
            BatchNorm2d(num_features=5),
            ReLU()
        )

        # Conv 3@ (1x1) stride 1 -> 6@ (2x2) stride 1 -> 9@ (4x4) stride 2
        self.convBlock2 = Sequential(                   
            Conv2d(kernel_size=1, stride=1, in_channels=1, out_channels=3),
            BatchNorm2d(num_features=3),
            ReLU(),

            Conv2d(kernel_size=2, stride=1, in_channels=3, out_channels=6, padding='same'),
            BatchNorm2d(num_features=6),
            ReLU(),

            Conv2d(kernel_size=4, stride=2, in_channels=6, out_channels=9, padding=1),
            BatchNorm2d(num_features=9),
            ReLU()
            #Dropout2d(0.1)
        )
        self.apply(self._init_weights)

        

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, x) -> torch.Tensor:
        x1 = self.block1(x)
        x2 = self.convBlock1(x)
        x3 = self.convBlock2(x)
        # print(x1.shape, x2.shape, x3.shape)
        y = torch.cat((x1, x2, x3), dim=1)
        return y

def z_score(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0

    for x, _ in loader:
        x = x.float()
        total_sum += x.sum().item()
        total_sq_sum += (x ** 2).sum().item()
        total_pixels += x.numel()

    mean_global = total_sum / total_pixels
    var_global = (total_sq_sum / total_pixels) - (mean_global ** 2)
    std_global = (var_global ** 0.5) + 1e-8
    
    return mean_global, std_global


if __name__ == "__main__":
    model = ConvolutionalRecurrentNet()
    train_dataset = CFR(folder="../data/doppler_traces/S1", campaigns=["a", "b"], split_mode="train", stride=25, transform=SpectogramAugmentation())
    val_dataset = CFR(folder="../data/doppler_traces/S1", campaigns=["c"], split_mode="val", stride=5)

    mean, std = z_score(train_dataset)
    train_transform = Compose([SpectogramAugmentation(), Normalize(mean, std)])
    val_transform = Normalize(mean, std)

    train_dataset = CFR(folder="../data/doppler_traces/S1", campaigns=["a", "b"], split_mode="train", stride=25, transform=train_transform)
    val_dataset = CFR(folder="../data/doppler_traces/S1", campaigns=["c"], split_mode="val", stride=5, transform=val_transform)

    batch_size = 32
    num_workers = 0
    pin_memory = torch.cuda.is_available()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory)

    valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)


    opt = Adam(model.parameters(), lr=1.5e-4, weight_decay = 3e-4)
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    epochs = 100
    patience = 15
    counter = 0

    best_val = np.inf
    checkpoint_path = "./models/recurrent4_model.pt"

    history = {
        "train": [],
        "val": [],
        "acc": []
    }

    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    for epoch in range(epochs):
        # TRAINING
        model.train()
        print(f"Epoch: {epoch+1}")

        cumtrain_loss = 0
        ntrain = 0
        train_iterator = tqdm(train_dataloader)
        for batch_x, batch_y in train_iterator:
            batch_x = batch_x.view(-1,1,340,100).to(device)
            batch_y = batch_y.repeat_interleave(4).to(device)

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)

            opt.zero_grad()
            loss.backward()
            # gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                size = batch_x.size(0)
                batch_x = batch_x.view(-1,1,340,100).to(device)
                batch_y = batch_y.to(device)

                y_pred = model(batch_x)
                y_pred_grouped = y_pred.view(size, 4, -1).mean(dim=1) # we average the predictions of the 4 windows to get a single prediction for each sample
                batch_loss = loss_fn(y_pred_grouped, batch_y)

                cumval_loss += batch_loss.item() * size
                nval += size

                predictions = y_pred_grouped.argmax(dim=1)
                nval_correct += (predictions == batch_y).sum().item()

                val_iterator.set_description(f"Validation loss: {batch_loss.item():.5f}")

            val_loss = cumval_loss / nval
            val_acc = nval_correct / nval
            history["val"].append(val_loss)
            history["acc"].append(val_acc)
            print(f"Validation loss: {val_loss}, accuracy: {val_acc}")

        # EARLY STOPPING
        if val_loss < best_val:
            print("Saved Model")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': val_loss,
                'train_mean': mean,
                'train_std': std
            }
            torch.save(checkpoint, checkpoint_path)
            best_val = val_loss
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print(f"[EARLY STOPPING] Validation loss hasn't improved for {patience} epochs.")
            break

        # temporal update of the learning rate:
        scheduler.step()


    history_path = "plot_data/training_history_recurrent4.json"

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)