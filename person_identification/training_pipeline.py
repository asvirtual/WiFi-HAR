import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import CFR_PersonID_4Channels as CFR_PersonID
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from models import InceptionCNN
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# 1. DATA AUGMENTATION MODULE
# ==========================================
class SpecAugment:
    
    def __init__(self, noise_std=0.02, scale_range=(0.8, 1.2), max_time_mask=30):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.max_time_mask = max_time_mask

    def __call__(self, x):
        # Random Scaling: Simulates variations in signal strength/distance
        scale_factor = torch.empty(1).uniform_(*self.scale_range).item()
        x = x * scale_factor
        
        # Time Masking: Randomly zeros out a time chunk to simulate packet loss or temporary occlusions
        if torch.rand(1).item() > 0.5:
            mask_len = torch.randint(10, self.max_time_mask, (1,)).item()
            t0 = torch.randint(0, max(1, x.shape[1] - mask_len), (1,)).item()
            x[:, t0:t0+mask_len, :] = 0.0
            
        # Gaussian Noise: Adds baseline thermal/hardware noise
        noise = torch.randn_like(x) * self.noise_std
        x = x + noise
        return x

# ==========================================
# 2. CONTRASTIVE LEARNING LOSS & VIEWS
# ==========================================
class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning (SupCon) Loss.
    Pulls together feature embeddings of the same class (even across different augmented views)
    while pushing apart embeddings of different classes.
    """
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features shape expected: [batch_size, views, embedding_dim]
        if features.dim() != 3:
            raise ValueError("features must have shape [batch_size, views, embedding_dim]")
        
        batch_size, view_count, embedding_dim = features.shape
        device = features.device
        
        # Normalize and flatten features
        features = F.normalize(features, dim=-1).reshape(batch_size * view_count, embedding_dim)
        labels = labels.view(-1, 1).repeat(1, view_count).reshape(-1, 1)
        
        # Create a mask to identify positive pairs (same class)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute dot-product similarity logits
        logits = torch.matmul(features, features.T) / self.temperature
        
        # Numerical stability trick (subtract max)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # Remove self-comparisons from the mask and logits
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        
        # Compute mean log-prob for positive pairs
        positives = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positives.clamp_min(1.0)
        
        return -mean_log_prob_pos.mean()

def build_contrastive_views(batch_x, transform, device, view_count=2):
    """
    Generates multiple augmented versions ('views') of a single input batch.
    Required for contrastive learning to teach the network invariance to transformations.
    """
    views = []
    for _ in range(view_count):
        augmented_samples = []
        for sample in batch_x:
            # Detach and move to CPU for augmentation processing
            sample = sample.detach().clone().cpu()
            if transform is not None:
                sample = transform(sample)
            augmented_samples.append(sample)
        # Stack back into a batch and move to target device (GPU)
        views.append(torch.stack(augmented_samples, dim=0).to(device))
    return views

# ==========================================
# 3. INITIALIZATION & DATASET SETUP
# ==========================================
model = InceptionCNN()
transform = SpecAugment(noise_std=0.02, scale_range=(0.8, 1.2), max_time_mask=30)


campaigns_train = [
    "S1a", "S4a",  # P1: 
    "S3a", "S5a",  # P2: 
    "S7a"          # P3: 
]

# Validation campaigns 
campaigns_val = [
    "S1b", "S1c",  # P1: 
    "S3a", "S5a",  # P2: 
    "S7a"          # P3: 
]

# Test campaigns: 
campaigns_test = [
    "S2a",         # P0: ZERO-SHOT 1 
    "S6a",         # P0: ZERO-SHOT 2 
    "S4b",         # P0: STRESS TEST 
    "S3a", "S5a",  # P1: 
    "S7a"          # P2: 
]

attivita_target = ["W","R"]

# Initialize Custom DataLoaders
train_dataset = CFR_PersonID(folder="../data/doppler_traces/", campaigns=campaigns_train, target_activities=attivita_target, split_mode="train")
val_dataset   = CFR_PersonID(folder="../data/doppler_traces/", campaigns=campaigns_val, target_activities=attivita_target, split_mode="val")
test_dataset  = CFR_PersonID(folder="../data/doppler_traces/", campaigns=campaigns_test, target_activities=attivita_target, split_mode="test")

batch_size = 32
num_workers = 0
pin_memory = torch.cuda.is_available() 

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

# ==========================================
# 4. TRAINING HYPERPARAMETERS & SETUP
# ==========================================
loss_fn = CrossEntropyLoss() 
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

epochs = 70
patience = 15 # Early stopping tolerance
counter = 0
best_val = np.inf
checkpoint_path = "best_model.pt"

# Optimizer and Learning Rate Scheduler 
opt = Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

# Loss functions for the hybrid framework
loss_ce_fn = CrossEntropyLoss()
loss_supcon_fn = SupervisedContrastiveLoss(temperature=0.2)

# Warm-up parameters for SupCon Loss 
target_contrastive_weight = 0.25
warmup_epochs = 1 

# ==========================================
# 5. MAIN TRAINING & VALIDATION LOOP
# ==========================================
best_val_f1 = 0.0
min_delta=0.01 
history = {"train": [], "val": [], "acc": [], "f1": []}

epochs_pbar = tqdm(range(epochs), desc="Training Progress", unit="epoch")

for epoch in epochs_pbar:
    
    # Calculate dynamic SupCon weight (Warm-up logic)
    current_contrastive_weight = target_contrastive_weight * min(1.0, epoch / warmup_epochs)
    
    # --- TRAINING PHASE ---
    model.train()
    cumtrain_loss = 0
    ntrain = 0

    for batch_x, batch_y in train_dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # 1. Generate multi-view augmented batch
        views = build_contrastive_views(batch_x, transform, device, view_count=2)
        
        view_logits = []
        view_projections = []
        
        # 2. Extract features and logits for each view
        for view in views:
            logits, proj = model(view, return_projection=True)
            view_logits.append(logits)
            view_projections.append(proj)

        projected_features = torch.stack(view_projections, dim=1)

        # 3. Calculate Hybrid Loss
        # CrossEntropy is averaged across all views
        loss_ce = torch.stack([loss_ce_fn(l, batch_y) for l in view_logits]).mean()
        # SupCon loss applies on the latent projections
        loss_supcon = loss_supcon_fn(projected_features, batch_y)
        
        total_loss = loss_ce + (current_contrastive_weight * loss_supcon)

        # 4. Backpropagation
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Prevents exploding gradients
        opt.step()

        cumtrain_loss += total_loss.item() * batch_x.size(0)
        ntrain += batch_x.size(0)

    train_loss = cumtrain_loss / ntrain
    history["train"].append(train_loss)

    # --- VALIDATION PHASE ---
    model.eval()
    cumval_loss = 0
    nval_correct = 0
    nval = 0
    
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch_x, batch_y in valid_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            
            y_pred = model(batch_x)
            batch_loss = loss_fn(y_pred, batch_y)

            cumval_loss += batch_loss.item() * batch_x.size(0)
            nval += batch_x.size(0)

            predictions = y_pred.argmax(dim=1)
            nval_correct += (predictions == batch_y).sum().item()
            
            val_preds.extend(predictions.cpu().numpy())
            val_targets.extend(batch_y.cpu().numpy())

    val_loss = cumval_loss / nval
    val_acc = nval_correct / nval
    
    # Calculate Macro F1 (better metric for imbalanced classes)
    val_macro_f1 = f1_score(val_targets, val_preds, average='macro')
    
    history["val"].append(val_loss)
    history["acc"].append(val_acc)
    history["f1"].append(val_macro_f1)

    # Update progress bar statistics
    epochs_pbar.set_postfix({
        "Train Loss": f"{train_loss:.4f}",
        "Val Loss": f"{val_loss:.4f}",
        "Val F1": f"{val_macro_f1*100:.2f}%",
        "SupCon Weight": f"{current_contrastive_weight:.3f}" # Monitor warm-up progression
    })

    # Model Checkpointing based on Macro F1 Score
    if val_macro_f1 > best_val_f1+min_delta:
        torch.save(model.state_dict(), checkpoint_path)
        best_val_f1 = val_macro_f1
        counter = 0 # Reset patience
    else:
        counter += 1

    # Early Stopping check
    if counter >= patience:
        epochs_pbar.write(f"\n[EARLY STOPPING] Validation Macro F1 is not improved after {patience} epochs.")
        break
        
    scheduler.step()

# ==========================================
# 6. TRAINING PLOTS & VISUALIZATIONS
# ==========================================
plt.figure(figsize=(12, 5))

# Plot 1: Loss curves
plt.subplot(1, 2, 1)
plt.plot(history["train"], label="Train Loss", color="blue", lw=2)
plt.plot(history["val"], label="Validation Loss", color="orange", lw=2)
plt.title("Loss behaviour")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Plot 2: Evaluation metrics
plt.subplot(1, 2, 2)
plt.plot(history["acc"], label="Val Accuracy", color="green", lw=2)
plt.plot(history["f1"], label="Val Macro F1", color="purple", lw=2, linestyle="--")
plt.title("EVALUATION METRICS")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()


print("\n---TEST SET EVALUATION---")
# Load the best model weights found during training
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

cumtest_loss = 0
ntest_correct = 0
ntest = 0

all_preds = []
all_targets = []

with torch.no_grad():
    for batch_x, batch_y in tqdm(test_dataloader, desc="Testing", leave=False, dynamic_ncols=True):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        y_pred = model(batch_x)
        batch_loss = loss_fn(y_pred, batch_y)

        cumtest_loss += batch_loss.item() * batch_x.size(0)
        ntest += batch_x.size(0)

        predictions = y_pred.argmax(dim=1)
        ntest_correct += (predictions == batch_y).sum().item()
        
        all_preds.extend(predictions.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())

test_loss = cumtest_loss / ntest
test_acc = ntest_correct / ntest

# Compute Global Macro F1 and class-specific F1 scores
test_macro_f1 = f1_score(all_targets, all_preds, average='macro')
class_f1_scores = f1_score(all_targets, all_preds, average=None)


test_conf = np.exp(-test_loss) * 100

print(f"Test Loss: {test_loss:.4f} | Confidenza Media: {test_conf:.2f}%")
print(f"Final Accuracy: {test_acc*100:.2f}% | Final Macro F1: {test_macro_f1*100:.2f}%\n")
print(f"F1-Score per classe -> P0: {class_f1_scores[0]*100:.2f}% | P1: {class_f1_scores[1]*100:.2f}% | P2: {class_f1_scores[2]*100:.2f}%")

# ==========================================
# 8. CONFUSION MATRIX VISUALIZATION
# ==========================================


cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2], normalize='true')
fig, ax = plt.subplots(figsize=(9, 7))

# Create custom labels showing individual F1 scores
display_labels = [
    f'P0\n[F1: {class_f1_scores[0]*100:.1f}%]', 
    f'P1\n[F1: {class_f1_scores[1]*100:.1f}%]', 
    f'P2\n[F1: {class_f1_scores[2]*100:.1f}%]'
]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(cmap='Blues', values_format='.1%', ax=ax)

plt.title(f"Confusion Matrix\nAcc: {test_acc*100:.1f}% | Macro F1: {test_macro_f1*100:.1f}%", pad=15)
plt.xlabel("PREDICTED LABEL", labelpad=10)
plt.ylabel("TRUE LABEL", labelpad=10)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()