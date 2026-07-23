import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import CFR_PersonID_4Channels as CFR_PersonID
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from models import BaselineNet
from sklearn.metrics import f1_score


# =====================================================================
# 1. DEFINIZIONE DATA AUGMENTATION (Risolve l'errore "transform non definita")
# =====================================================================
class DopplerAugmentation:
    def __init__(self, noise_std=0.02, scale_range=(0.8, 1.2), max_time_mask=30):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.max_time_mask = max_time_mask

    def __call__(self, x):
        # 1. Random Scaling
        scale_factor = torch.empty(1).uniform_(*self.scale_range).item()
        x = x * scale_factor
        
        # 2. Time Masking (Simula perdita pacchetti)
        if torch.rand(1).item() > 0.5:
            mask_len = torch.randint(10, self.max_time_mask, (1,)).item()
            t0 = torch.randint(0, max(1, x.shape[1] - mask_len), (1,)).item()
            x[:, t0:t0+mask_len, :] = 0.0
            
        # 3. Gaussian Noise
        noise = torch.randn_like(x) * self.noise_std
        x = x + noise
        return x

# =====================================================================
# 2. DEFINIZIONE LOSS CONTRASTIVA E VISTE
# =====================================================================
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        if features.dim() != 3:
            raise ValueError("features must have shape [batch_size, views, embedding_dim]")
        batch_size, view_count, embedding_dim = features.shape
        device = features.device
        features = F.normalize(features, dim=-1).reshape(batch_size * view_count, embedding_dim)
        labels = labels.view(-1, 1).repeat(1, view_count).reshape(-1, 1)
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

def build_contrastive_views(batch_x, transform, device, view_count=2):
    views = []
    for _ in range(view_count):
        augmented_samples = []
        for sample in batch_x:
            sample = sample.detach().clone().cpu()
            if transform is not None:
                sample = transform(sample)
            augmented_samples.append(sample)
        views.append(torch.stack(augmented_samples, dim=0).to(device))
    return views

# =====================================================================
# 3. CONFIGURAZIONE DATASET E MODELLO
# =====================================================================
model = BaselineNet()

# Istanziamo la trasformazione qui, pronta per essere usata nel training!
transform = DopplerAugmentation(noise_std=0.02, scale_range=(0.8, 1.2), max_time_mask=30)



# 1. TRAINING SET (La palestra di invarianza)

campagne_train = [
    "S1a", "S4a",  # P0: 
    "S3a", "S5a",  # P1: 
    "S7a"          # P2: 
]

# 2. VALIDATION SET (Termometro pulito per l'Early Stopping)
campagne_val = [
    "S1b", "S1c",  # P0: 
    "S3a", "S5a",  # P1: 
    "S7a"          # P2: 
]

# 3. TEST SET (L'esame finale: Zero-Shot su stanze nuove + Stress Test)
# Selezioniamo un numero mirato di sessioni per P0 per NON sbilanciare la Matrice.
campagne_test = [
    "S2a",         # P0: ZERO-SHOT 1 
    "S6a",         # P0: ZERO-SHOT 2 
    "S4b",         # P0: STRESS TEST 
    "S3a", "S5a",  # P1: 
    "S7a"          # P2: 
]

attivita_target = ["W","R"]

train_dataset = CFR_PersonID(folder="../data/doppler_traces/", campaigns=campagne_train, target_activities=attivita_target, split_mode="train")
val_dataset   = CFR_PersonID(folder="../data/doppler_traces/", campaigns=campagne_val, target_activities=attivita_target, split_mode="val")
test_dataset  = CFR_PersonID(folder="../data/doppler_traces/", campaigns=campagne_test, target_activities=attivita_target, split_mode="test")

batch_size = 32
num_workers = 0
pin_memory = torch.cuda.is_available()

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

opt = Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
loss_fn = CrossEntropyLoss() # Usata per la validazione standard
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

epochs = 70
patience = 15
counter = 0
best_val = np.inf
checkpoint_path = "best_model.pt"

history = {"train": [], "val": [], "acc": []}

loss_ce_fn = CrossEntropyLoss()
loss_supcon_fn = SupervisedContrastiveLoss(temperature=0.2)
contrastive_weight = 0.25

from sklearn.metrics import f1_score

# =====================================================================
# 4. TRAINING & VALIDATION LOOP
# =====================================================================
# Inizializziamo il best value a 0.0 perché vogliamo MASSIMIZZARE l'F1-Score!
best_val_f1 = 0.0  
history = {"train": [], "val": [], "acc": [], "f1": []}

epochs_pbar = tqdm(range(epochs), desc="Training Progress", unit="epoch")

for epoch in epochs_pbar:
    # --- TRAINING ---
    model.train()
    cumtrain_loss = 0
    ntrain = 0

    for batch_x, batch_y in train_dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Creazione delle viste per la Contrastive Loss
        views = build_contrastive_views(batch_x, transform, device, view_count=2)
        
        view_logits = []
        view_projections = []
        
        for view in views:
            logits, proj = model(view, return_projection=True)
            view_logits.append(logits)
            view_projections.append(proj)

        projected_features = torch.stack(view_projections, dim=1)

        loss_ce = torch.stack([loss_ce_fn(l, batch_y) for l in view_logits]).mean()
        loss_supcon = loss_supcon_fn(projected_features, batch_y)
        
        total_loss = loss_ce + (contrastive_weight * loss_supcon)

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        cumtrain_loss += total_loss.item() * batch_x.size(0)
        ntrain += batch_x.size(0)

    train_loss = cumtrain_loss / ntrain
    history["train"].append(train_loss)

    # --- VALIDATION ---
    model.eval()
    cumval_loss = 0
    nval_correct = 0
    nval = 0
    
    # Liste per accumulare predizioni e target per calcolare l'F1-Score
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
            
            # Salviamo per il calcolo del Macro F1
            val_preds.extend(predictions.cpu().numpy())
            val_targets.extend(batch_y.cpu().numpy())

    val_loss = cumval_loss / nval
    val_acc = nval_correct / nval
    
    # CALCOLO DEL MACRO F1 DI VALIDAZIONE
    val_macro_f1 = f1_score(val_targets, val_preds, average='macro')
    
    history["val"].append(val_loss)
    history["acc"].append(val_acc)
    history["f1"].append(val_macro_f1)

    epochs_pbar.set_postfix({
        "Train Loss": f"{train_loss:.4f}",
        "Val Loss": f"{val_loss:.4f}",
        "Val Acc": f"{val_acc*100:.2f}%",
        "Val F1": f"{val_macro_f1*100:.2f}%"
    })

    # NUOVA LOGICA DI SALVATAGGIO: Salva quando il Macro F1-Score è MASSIMO!
    if val_macro_f1 > best_val_f1:
        torch.save(model.state_dict(), checkpoint_path)
        best_val_f1 = val_macro_f1
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        epochs_pbar.write(f"\n[EARLY STOPPING] Macro F1 di validazione non migliora da {patience} epoche.")
        break


# =====================================================================
# 5. VALUTAZIONE FINALE E GRAFICI
# =====================================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history["train"], label="Train Loss", color="blue", lw=2)
plt.plot(history["val"], label="Validation Loss", color="orange", lw=2)
plt.title("Andamento della Loss")
plt.xlabel("Epoche")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history["acc"], label="Val Accuracy", color="green", lw=2)
plt.plot(history["f1"], label="Val Macro F1", color="purple", lw=2, linestyle="--")
plt.title("Metriche di Validazione")
plt.xlabel("Epoche")
plt.ylabel("Score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

print("\n--- Valutazione sul Test Set ---")
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
test_macro_f1 = f1_score(all_targets, all_preds, average='macro')
test_conf = np.exp(-test_loss) * 100

print(f"Test Loss: {test_loss:.4f}")
print(f"Confidenza Media: {test_conf:.2f}% | Accuracy Finale: {test_acc*100:.2f}% | Macro F1 Finale: {test_macro_f1*100:.2f}%\n")

cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2])
fig, ax = plt.subplots(figsize=(8, 6))
display_labels = ['Persona 0 (P0)', 'Persona 1 (P1)', 'Persona 2 (P2)']

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(cmap='Blues', values_format='d', ax=ax)

plt.title(f"Matrice di Confusione (Test Acc: {test_acc*100:.1f}% | F1: {test_macro_f1*100:.1f}%)", pad=15)
plt.xlabel("Etichetta Predetta (Modello)")
plt.ylabel("Etichetta Reale (Ground Truth)")
plt.tight_layout()

plt.savefig("confusion_matrix_test.png", dpi=300)
print("--> Grafico salvato come 'confusion_matrix_test.png'")
plt.show()