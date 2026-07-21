import json
import torch
import matplotlib.pyplot as plt
from torch.nn import LSTM, BatchNorm1d, InstanceNorm2d, Conv2d, MaxPool2d, ReLU, Dropout, Sequential, Linear, Flatten, CrossEntropyLoss, Tanh, Parameter, TransformerEncoderLayer, TransformerEncoder
import numpy as np
from torch.utils.data import DataLoader
from dataset2 import CFR, SpectogramAugmentation
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from tqdm import tqdm

class ConvolutionalTransformerNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = Sequential(
            InceptionModule(),
            Conv2d(kernel_size=4,stride=2,padding=0,out_channels=32,in_channels=15),  # 32, 84, 24
            InstanceNorm2d(num_features=32, affine=True), 
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2), # 32, 42, 12
        )

        self.pos_embedding = Parameter(torch.randn(1,42,384)*0.01)
        encoder_layer = TransformerEncoderLayer(d_model=32*12, nhead=4, dim_feedforward=512, dropout=0.2)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)

        self.classificator=Sequential(
            Dropout(0.2),
            Linear(in_features=384*2, out_features=128), # head projection that maps features from the LSTM (bidirectional) to 128 features that merge those informations (we have 256 since we also keep the standar deviation)
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
        x  = x + self.pos_embedding
        x = self.transformer(x)

        mean = x.mean(dim=1)
        std = torch.std(x, dim=1, unbiased=False)
        x = torch.cat((mean, std), dim=-1)
        return self.classificator(x)


class InceptionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Max Pool
        self.block1 = MaxPool2d(kernel_size=2, stride=2)
        
        # Conv @5 (2x2) stride 2
        self.convBlock1 = Sequential(
            Conv2d(kernel_size=2, stride=2, out_channels=5, in_channels=1),
            InstanceNorm2d(num_features=5, affine=True),
            ReLU()
        )

        # Conv 3@ (1x1) stride 1 -> 6@ (2x2) stride 1 -> 9@ (4x4) stride 2
        self.convBlock2 = Sequential(                   
            Conv2d(kernel_size=1, stride=1, in_channels=1, out_channels=3),
            InstanceNorm2d(num_features=3, affine=True),
            ReLU(),

            Conv2d(kernel_size=2, stride=1, in_channels=3, out_channels=6, padding='same'),
            InstanceNorm2d(num_features=6, affine=True),
            ReLU(),

            Conv2d(kernel_size=4, stride=2, in_channels=6, out_channels=9, padding=1),
            InstanceNorm2d(num_features=9, affine=True),
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


class SelfAttention(torch.nn.Module):
    def __init__(self, in_features, attention_dim):
        super().__init__()
        self.attention = Sequential(
            Linear(in_features=in_features, out_features=attention_dim),
            Tanh(),
            Linear(in_features=attention_dim, out_features=1, bias = False)
        )

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    
    def forward(self, x):
        scores = self.attention(x)  # (batch_size, time_steps, 1)
        weights = torch.softmax(scores, dim=1) # normalize scores with softmax
        context = torch.sum(weights * x, dim=1)  # weighted sum of the LSTM outputs: (batch_size, features)
        return context, weights




if __name__ == "__main__":
    model = ConvolutionalTransformerNet()
    transform = SpectogramAugmentation()
    train_dataset = CFR(folder="../data/doppler_traces/S1", campaigns=["a", "b"], split_mode="train", stride=25, transform=transform)
    val_dataset = CFR(folder="../data/doppler_traces/S1", campaigns=["c"], split_mode="val", stride=5)

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
    checkpoint_path = "./models/transformer_model.pt"

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

        # temporal update of the learning rate:
        scheduler.step()


    history_path = "plot_data/training_history_transformer.json"

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)