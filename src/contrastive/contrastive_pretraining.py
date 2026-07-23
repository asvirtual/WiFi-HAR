import json
import os
import torch
import matplotlib.pyplot as plt
from torch.nn import BatchNorm1d, InstanceNorm2d, Conv2d, MaxPool2d, ReLU, Dropout, Sequential, Linear, Flatten, CrossEntropyLoss
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataset import CFR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from tqdm import tqdm


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


def set_module_trainable(module, trainable):
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def forward_multiview_logits(model, batch_x):
    view_logits = []
    for view_idx in range(batch_x.size(1)):
        view_logits.append(model(batch_x[:, view_idx]))
    return torch.stack(view_logits, dim=0).mean(dim=0)


def forward_multiview_projections(model, batch_x):
    view_features = []
    for view_idx in range(batch_x.size(1)):
        _, features = model(batch_x[:, view_idx], return_features=True)
        view_features.append(model.project(features))
    return torch.stack(view_features, dim=1)


def train_contrastive_epoch(model, dataloader, optimizer, contrastive_loss_fn, device):
    model.train()
    set_module_trainable(model.backbone, True)
    set_module_trainable(model.projector, True)
    set_module_trainable(model.classifier, False)

    cumulative_loss = 0
    sample_count = 0

    train_iterator = tqdm(dataloader)
    for batch_x, batch_y in train_iterator:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        projected_features = forward_multiview_projections(model, batch_x)
        loss = contrastive_loss_fn(projected_features, batch_y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.backbone.parameters()) + list(model.projector.parameters()), max_norm=1.0)
        optimizer.step()

        cumulative_loss += loss.item() * batch_x.size(0)
        sample_count += batch_x.size(0)
        train_iterator.set_description(f"Pretrain loss: {loss.item():.5f}")

    return cumulative_loss / sample_count


def evaluate_contrastive_epoch(model, dataloader, contrastive_loss_fn, device):
    model.eval()
    cumulative_loss = 0
    sample_count = 0

    with torch.no_grad():
        val_iterator = tqdm(dataloader)
        for batch_x, batch_y in val_iterator:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            projected_features = forward_multiview_projections(model, batch_x)
            loss = contrastive_loss_fn(projected_features, batch_y)

            cumulative_loss += loss.item() * batch_x.size(0)
            sample_count += batch_x.size(0)
            val_iterator.set_description(f"Pretrain val loss: {loss.item():.5f}")

    return cumulative_loss / sample_count


def train_finetune_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    set_module_trainable(model.backbone, True)
    set_module_trainable(model.projector, False)
    set_module_trainable(model.classifier, True)

    cumulative_loss = 0
    sample_count = 0

    train_iterator = tqdm(dataloader)
    for batch_x, batch_y in train_iterator:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = forward_multiview_logits(model, batch_x)
        loss = loss_fn(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.backbone.parameters()) + list(model.classifier.parameters()), max_norm=1.0)
        optimizer.step()

        cumulative_loss += loss.item() * batch_x.size(0)
        sample_count += batch_x.size(0)
        train_iterator.set_description(f"Finetune loss: {loss.item():.5f}")

    return cumulative_loss / sample_count


def evaluate_finetune_epoch(model, dataloader, loss_fn, device):
    model.eval()
    cumulative_loss = 0
    correct_predictions = 0
    sample_count = 0

    with torch.no_grad():
        val_iterator = tqdm(dataloader)
        for batch_x, batch_y in val_iterator:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = forward_multiview_logits(model, batch_x)
            loss = loss_fn(logits, batch_y)

            cumulative_loss += loss.item() * batch_x.size(0)
            sample_count += batch_x.size(0)
            correct_predictions += (logits.argmax(dim=1) == batch_y).sum().item()
            val_iterator.set_description(f"Finetune val loss: {loss.item():.5f}")

    return cumulative_loss / sample_count, correct_predictions / sample_count


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

if __name__ == "__main__":
    model = BaselineNet()
    train_dataset = CFR(folder="../../data/doppler_traces/S1", campaigns=["a", "b"], split_mode="train", stride=5, transform=None, use_multi_antenna=True)
    val_dataset = CFR(folder="../../data/doppler_traces/S1", campaigns=["c"], split_mode="val", stride=5, transform=None, use_multi_antenna=True)

    batch_size = 64
    num_workers = 0
    pin_memory = torch.cuda.is_available()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory)

    valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)
    contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    pretrain_epochs = 20
    finetune_epochs = 30
    patience = 10
    counter = 0

    pretrain_best_val = np.inf
    finetune_best_val = np.inf
    pretrain_checkpoint_path = "./models/contrastive_encoder_pretrain.pt"
    checkpoint_path = "./models/contrastive_best_checkpoint.pt"

    history = {
        "pretrain": {
            "train": [],
            "val": [],
        },
        "finetune": {
            "train": [],
            "val": [],
            "acc": [],
        }
    }

    pretrain_optimizer = Adam(list(model.backbone.parameters()) + list(model.projector.parameters()), lr=3e-4, weight_decay=1e-4)
    finetune_optimizer = Adam(list(model.backbone.parameters()) + list(model.classifier.parameters()), lr=3e-4, weight_decay=1e-4)

    pretrain_scheduler = CosineAnnealingLR(pretrain_optimizer, T_max=pretrain_epochs, eta_min=1e-6)
    finetune_scheduler = CosineAnnealingLR(finetune_optimizer, T_max=finetune_epochs, eta_min=1e-6)

    for epoch in range(pretrain_epochs):
        print(f"[PRETRAIN] Epoch: {epoch + 1}")
        train_loss = train_contrastive_epoch(model, train_dataloader, pretrain_optimizer, contrastive_loss_fn, device)
        val_loss = evaluate_contrastive_epoch(model, valid_dataloader, contrastive_loss_fn, device)
        history["pretrain"]["train"].append(train_loss)
        history["pretrain"]["val"].append(val_loss)
        print(f"[PRETRAIN] train loss: {train_loss}, val loss: {val_loss}")

        if val_loss < pretrain_best_val:
            print("[PRETRAIN] Saved encoder checkpoint")
            save_checkpoint(
                checkpoint_path=pretrain_checkpoint_path,
                model=model,
                optimizer=pretrain_optimizer,
                epoch=epoch + 1,
                best_val=val_loss,
                history=history,
                extra_metadata={
                    "stage": "pretrain",
                    "use_multi_antenna": True,
                    "temperature": contrastive_loss_fn.temperature,
                },
            )
            pretrain_best_val = val_loss

        pretrain_scheduler.step()

    load_checkpoint(pretrain_checkpoint_path, model, optimizer=pretrain_optimizer, map_location=device)

    for parameter in model.projector.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    counter = 0
    for epoch in range(finetune_epochs):
        print(f"[FINETUNE] Epoch: {epoch + 1}")
        train_loss = train_finetune_epoch(model, train_dataloader, finetune_optimizer, loss_fn, device)
        val_loss, val_acc = evaluate_finetune_epoch(model, valid_dataloader, loss_fn, device)
        history["finetune"]["train"].append(train_loss)
        history["finetune"]["val"].append(val_loss)
        history["finetune"]["acc"].append(val_acc)
        print(f"[FINETUNE] train loss: {train_loss}, val loss: {val_loss}, accuracy: {val_acc}")

        if val_loss < finetune_best_val:
            print("[FINETUNE] Saved model")
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=finetune_optimizer,
                epoch=epoch + 1,
                best_val=val_loss,
                history=history,
                extra_metadata={
                    "stage": "finetune",
                    "use_multi_antenna": True,
                    "pretrain_checkpoint_path": pretrain_checkpoint_path,
                },
            )
            finetune_best_val = val_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"[EARLY STOPPING] Validation loss hasn't improved for {patience} epochs.")
            break

        finetune_scheduler.step()


    history_path = "plot_data/training_history_baseline3.json"

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)