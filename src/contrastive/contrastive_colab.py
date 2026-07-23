import json
import os
import torch
import numpy as np
from torch.nn import BatchNorm1d, InstanceNorm2d, Conv2d, MaxPool2d, ReLU, Dropout, Sequential, Linear, Flatten, CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from tqdm import tqdm

from dataset_colab import CFR


class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        if features.dim() != 3:
            raise ValueError("features must have shape [batch_size, views, embedding_dim]")

        batch_size, view_count, embedding_dim = features.shape
        device = features.device

        features = F.normalize(features, dim=-1).reshape(batch_size * view_count, embedding_dim)

        labels = labels.view(-1, 1)
        labels = labels.repeat(1, view_count).reshape(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positives = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positives.clamp_min(1.0)

        return -mean_log_prob_pos.mean()


class ContrastiveBaselineNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Sequential(
            InceptionModule(),
            Conv2d(kernel_size=4, stride=2, padding=0, out_channels=32, in_channels=15),
            InstanceNorm2d(num_features=32, affine=True),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            Dropout(0.2),
            Linear(in_features=32 * 42 * 12, out_features=128),
            ReLU(),
            BatchNorm1d(num_features=128, momentum=0.01),
            Dropout(0.1),
        )
        self.classifier = Linear(in_features=128, out_features=8)
        self.projector = Sequential(
            Linear(in_features=128, out_features=128),
            ReLU(),
            Linear(in_features=128, out_features=64),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def encode(self, x):
        return self.backbone(x)

    def project(self, x):
        return self.projector(x)

    def classify(self, x):
        return self.classifier(x)

    def forward(self, x, return_features=False, return_projection=False):
        features = self.encode(x)
        logits = self.classify(features)
        if return_projection:
            return logits, self.project(features)
        if return_features:
            return logits, features
        return logits


BaselineNet = ContrastiveBaselineNet


class InceptionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = MaxPool2d(kernel_size=2, stride=2)
        self.convBlock1 = Sequential(
            Conv2d(kernel_size=2, stride=2, out_channels=5, in_channels=1),
            InstanceNorm2d(num_features=5, affine=True),
            ReLU(),
        )
        self.convBlock2 = Sequential(
            Conv2d(kernel_size=1, stride=1, in_channels=1, out_channels=3),
            InstanceNorm2d(num_features=3, affine=True),
            ReLU(),
            Conv2d(kernel_size=2, stride=1, in_channels=3, out_channels=6, padding='same'),
            InstanceNorm2d(num_features=6, affine=True),
            ReLU(),
            Conv2d(kernel_size=4, stride=2, in_channels=6, out_channels=9, padding=1),
            InstanceNorm2d(num_features=9, affine=True),
            ReLU(),
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
        return torch.cat((x1, x2, x3), dim=1)


def get_contrastive_weight(epoch, target_weight, warmup_epochs):
    if warmup_epochs <= 0:
        return target_weight
    warmup_progress = min(1.0, (epoch + 1) / warmup_epochs)
    return target_weight * warmup_progress


def save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val, history, extra_metadata=None):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "best_val": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }
    if extra_metadata is not None:
        checkpoint["metadata"] = extra_metadata
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, map_location=None):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


if __name__ == "__main__":
    model = BaselineNet()
    train_dataset = CFR(folder="/content/data/doppler_traces/S1", campaigns=["a", "b"], split_mode="train", stride=5, transform=None, use_multi_antenna=True)
    val_dataset = CFR(folder="/content/data/doppler_traces/S1", campaigns=["c"], split_mode="val", stride=5, transform=None, use_multi_antenna=True)

    batch_size = 64
    num_workers = 4
    pin_memory = torch.cuda.is_available()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)

    opt = Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)
    contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    epochs = 50
    patience = 10
    counter = 0
    contrastive_weight = 0.15
    contrastive_warmup_epochs = 3

    best_val = np.inf
    checkpoint_path = "/content/models/contrastive_best_checkpoint.pt"

    history = {
        "train": [],
        "val": [],
        "acc": [],
        "ce": [],
        "contrastive": [],
        "total": [],
    }

    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    for epoch in range(epochs):
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

            view_logits = []
            view_features = []
            for view_idx in range(batch_x.size(1)):
                logits, features = model(batch_x[:, view_idx], return_features=True)
                view_logits.append(logits)
                view_features.append(model.project(features))

            stacked_logits = torch.stack(view_logits, dim=0)
            mean_logits = stacked_logits.mean(dim=0)
            classification_loss = loss_fn(mean_logits, batch_y)

            projected_features = torch.stack(view_features, dim=1)
            contrastive_loss = contrastive_loss_fn(projected_features, batch_y)
            current_contrastive_weight = get_contrastive_weight(epoch, contrastive_weight, contrastive_warmup_epochs)
            loss = classification_loss + current_contrastive_weight * contrastive_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            cumtrain_loss += loss.item() * batch_x.size(0)
            cumtrain_ce += classification_loss.item() * batch_x.size(0)
            cumtrain_contrastive += contrastive_loss.item() * batch_x.size(0)
            ntrain += batch_x.size(0)
            train_iterator.set_description(f"Train total: {loss.item():.5f}")

        history["train"].append(cumtrain_loss / ntrain)
        history["ce"].append(cumtrain_ce / ntrain)
        history["contrastive"].append(cumtrain_contrastive / ntrain)
        history["total"].append(cumtrain_loss / ntrain)

        model.eval()
        cumval_loss = 0
        nval_correct = 0
        nval = 0

        with torch.no_grad():
            val_iterator = tqdm(valid_dataloader)
            for batch_x, batch_y in val_iterator:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                val_logits = []
                for view_idx in range(batch_x.size(1)):
                    val_logits.append(model(batch_x[:, view_idx]))
                y_pred = torch.stack(val_logits, dim=0).mean(dim=0)
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
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=opt,
                epoch=epoch + 1,
                best_val=val_loss,
                history=history,
                extra_metadata={
                    "contrastive_weight": contrastive_weight,
                    "current_contrastive_weight": current_contrastive_weight,
                    "contrastive_warmup_epochs": contrastive_warmup_epochs,
                    "use_multi_antenna": True,
                    "temperature": contrastive_loss_fn.temperature,
                },
            )
            best_val = val_loss
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print(f"[EARLY STOPPING] Validation loss hasn't improved for {patience} epochs.")
            break

        scheduler.step()

    os.makedirs("/content/plot_data", exist_ok=True)
    history_path = "/content/plot_data/training_history_baseline3.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
