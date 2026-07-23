import json
import torch
import torch.nn.functional as F
from torch.nn import LSTM, BatchNorm1d, InstanceNorm2d, Conv2d, MaxPool2d, ReLU, Dropout, Sequential, Linear, Flatten, CrossEntropyLoss, Tanh
import numpy as np
from torch.utils.data import DataLoader
from dataset2 import CFR, SpectogramAugmentation
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from tqdm import tqdm

class ConvolutionalRecurrentNet(torch.nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.cnn  = Sequential(
            InceptionModule(),
            Conv2d(kernel_size=4,stride=2,padding=0,out_channels=32,in_channels=17),  # 32, 84, 24
            InstanceNorm2d(num_features=32, affine=True), 
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2), # 32, 42, 12
        )
        # we will treat the 42 time steps as a sequence, and each time step has 32*12 features (number of filter * number of frequency bins)
        # now we have double the parameters since we also want to deal with the derivative of the values in following time instances
        self.lstm = LSTM(input_size=32*12*2, hidden_size=96, num_layers=2, dropout=0.25, batch_first=True, bidirectional=True) 
        self.attention = SelfAttention(in_features=192, attention_dim=96)

        self.classificator=Sequential(
            Dropout(0.35),
            Linear(in_features=384, out_features=96), # head projection that maps features from the LSTM (bidirectional) to 128 features that merge those informations (we have 256 since we also keep the standar deviation)
            ReLU(),
            BatchNorm1d(num_features=96, momentum=0.01),
            Dropout(0.35),
            Linear(in_features=96,out_features=num_classes),
        )
        self.projection_head = Sequential(
            Linear(in_features=384, out_features=128),
            ReLU(),
            Linear(in_features=128, out_features=64),
        )
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    
    def forward(self,x, return_embedding=False, return_projection=False):
        # convolutional network
        x = self.cnn(x)
        # we reshape the output of the CNN to be suitable for the LSTM: (batch_size, time_steps, channels * features)
        x = x.permute(0, 2, 1, 3)
        batch_size, time_steps, channels, features = x.size()
        x = x.reshape(batch_size, time_steps, channels * features)

        # compute the derivative to pass to the lstm together with the output maps of the cnn
        delta_x = torch.zeros_like(x)
        delta_x[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :] # S_t - S_{t-1}
        x = torch.cat((x,delta_x), dim=-1)
        # recurrent layer
        x  = self.lstm(x)[0] # we only take the output of the last layer of the LSTM

        context, attention_weights = self.attention(x) # we apply the self-attention mechanism to get a context vector of size 128
        std = torch.std(x, dim=1, unbiased=False) # we take the std over the time dimension
        self.latest_attention_weights = attention_weights
        features = torch.cat((context,std), dim=-1) # we concatenate the mean, max and std to get a vector of size 128*3=384
        logits = self.classificator(features)
        if return_projection:
            return logits, self.projection_head(features), features
        if return_embedding:
            return logits, features
        return logits


class InceptionModule(torch.nn.Module):
    def __init__(self, use_mixstyle=True):
        super().__init__()
        # Max Pool
        self.block1 = Sequential(
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=1, out_channels=3, kernel_size=1),
            InstanceNorm2d(num_features=3, affine=True),
            ReLU()
        )
        
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
        self.use_mixstyle = use_mixstyle
        if self.use_mixstyle:
            self.mixstyle = MixStyle(p=0.5, alpha=0.1)
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
        if self.use_mixstyle:
            y = self.mixstyle(y)
        return y


class SelfAttention(torch.nn.Module):
    def __init__(self, in_features, attention_dim):
        super().__init__()
        self.attention = Sequential(
            Linear(in_features=in_features, out_features=attention_dim),
            Tanh(),
            Linear(in_features=attention_dim, out_features=1, bias = False)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    
    def forward(self, x):
        scores = self.attention(x)  # (batch_size, time_steps, 1)
        weights = torch.softmax(scores, dim=1) # normalize scores with softmax
        context = torch.sum(weights * x, dim=1)  # weighted sum of the LSTM outputs: (batch_size, features)
        return context, weights


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=0.5, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1.0 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size, view_count, embedding_dim = features.shape
        features = F.normalize(features, dim=-1)
        features = features.reshape(batch_size * view_count, embedding_dim)

        labels = labels.unsqueeze(1).repeat(1, view_count).reshape(-1)
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(features.device)

        logits = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positives = mask.sum(dim=1).clamp_min(1.0)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positives
        return -mean_log_prob_pos.mean()


# causally mix the style of 2 different samples in a batch during training, in this way we simulate the execution in a virtual environment we never saw before
class MixStyle(torch.nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):    
        if not self.training or torch.rand(1).item() > self.p:
            return x

        B, C, H, W = x.size()

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = torch.sqrt(var + self.eps)

        x_norm = (x - mu) / sig

        perm = torch.randperm(B)
        mu2, sig2 = mu[perm], sig[perm]

        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)

        mu_mix = lmda * mu + (1 - lmda) * mu2
        sig_mix = lmda * sig + (1 - lmda) * sig2

        return x_norm * sig_mix + mu_mix


def compute_class_weights(train_dataset, num_classes=5):
    labels = [int(entry[2].item()) for entry in train_dataset.window_info]
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    return weights / weights.mean()


def get_contrastive_weight(epoch, target_weight, warmup_epochs):
    if warmup_epochs <= 0:
        return target_weight
    warmup_progress = min(1.0, (epoch + 1) / warmup_epochs)
    return target_weight * warmup_progress


if __name__ == "__main__":
    model = ConvolutionalRecurrentNet()
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


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    class_weights = compute_class_weights(train_dataset, num_classes=5).to(device)
    opt = AdamW(model.parameters(), lr=5e-5, weight_decay=5e-4)
    ce_loss_fn = CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    focal_loss_fn = FocalLoss(gamma=0.5)
    contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.2)

    epochs = 100
    patience = 40
    counter = 0
    contrastive_weight = 0.1
    contrastive_warmup_epochs = 3

    best_val = np.inf
    best_val_acc = -np.inf
    checkpoint_path = "./models/contrastive_model.pt"
    window_size = 5
    val_loss_window = []
    val_acc_window = []

    history = {
        "train": [],
        "val": [],
        "acc": [],
        "ce": [],
        "contrastive": [],
        "total": []
    }

    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    for epoch in range(epochs):
        # TRAINING
        model.train()
        print(f"Epoch: {epoch+1}")

        cumtrain_loss = 0
        cumtrain_ce = 0
        cumtrain_contrastive = 0
        ntrain = 0
        train_iterator = tqdm(train_dataloader)
        for batch_x, batch_y in train_iterator:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            batch_x_flat = batch_x.reshape(-1, 1, 340, 100)
            logits, projection, _ = model(batch_x_flat, return_projection=True)
            logits = logits.view(batch_y.size(0), 4, -1)
            projection = projection.view(batch_y.size(0), 4, -1)
            logits_mean = logits.mean(dim=1)
            loss_ce = ce_loss_fn(logits_mean, batch_y)

            contrastive_loss = contrastive_loss_fn(projection, batch_y)
            current_contrastive_weight = get_contrastive_weight(
                epoch=epoch,
                target_weight=contrastive_weight,
                warmup_epochs=contrastive_warmup_epochs,
            )
            loss = loss_ce + current_contrastive_weight * contrastive_loss

            opt.zero_grad()
            loss.backward()
            # gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            cumtrain_loss += loss.item() * batch_y.size(0)
            cumtrain_ce += loss_ce.item() * batch_y.size(0)
            cumtrain_contrastive += contrastive_loss.item() * batch_y.size(0)
            ntrain += batch_y.size(0)
            train_iterator.set_description(f"Train loss: {loss.item():.5f}")

        history["train"].append(cumtrain_loss / ntrain)
        history["ce"].append(cumtrain_ce / ntrain)
        history["contrastive"].append(cumtrain_contrastive / ntrain)
        history["total"].append(cumtrain_loss / ntrain)

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

                batch_x_flat = batch_x.reshape(-1, 1, 340, 100)
                logits, _, _ = model(batch_x_flat, return_projection=True)
                logits = logits.view(batch_y.size(0), 4, -1)
                logits_mean = logits.mean(dim=1)
                batch_loss = ce_loss_fn(logits_mean, batch_y)
                cumval_loss += batch_loss.item() * batch_y.size(0)
                nval += batch_y.size(0)

                predictions = logits_mean.argmax(dim=1)
                nval_correct += (predictions == batch_y).sum().item()

                val_iterator.set_description(f"Validation loss: {batch_loss.item():.5f}")

            val_loss = cumval_loss / nval
            val_acc = nval_correct / nval
            history["val"].append(val_loss)
            history["acc"].append(val_acc)
            print(f"Validation loss: {val_loss}, accuracy: {val_acc}")

            val_loss_window.append(val_loss)
            val_acc_window.append(val_acc)
            if len(val_loss_window) > window_size:
                val_loss_window.pop(0)
                val_acc_window.pop(0)

        # EARLY STOPPING with rolling window
        if len(val_loss_window) == window_size:
            window_mean_loss = float(np.mean(val_loss_window))
            window_mean_acc = float(np.mean(val_acc_window))

            improved = False
            if window_mean_loss < best_val - 1e-6:
                improved = True
            elif abs(window_mean_loss - best_val) <= 1e-6 and window_mean_acc > best_val_acc + 1e-6:
                improved = True

            if improved:
                print("Saved Model")
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'loss': val_loss,
                    'accuracy': val_acc
                }
                torch.save(checkpoint, checkpoint_path)
                best_val = window_mean_loss
                best_val_acc = window_mean_acc
                counter = 0
            else:
                counter += 1
        else:
            counter += 1

        if counter >= patience:
            print(f"[EARLY STOPPING] Validation trend hasn't improved for {patience} epochs.")
            break


        # temporal update of the learning rate:
        scheduler.step(val_loss)


    history_path = "plot_data/training_history_contrastive.json"

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)


# possible later improvements -> implement softmax weighted entropy for the classification part
# take into consideration also the focal loss