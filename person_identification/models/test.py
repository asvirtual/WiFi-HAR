import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CFR_PersonID_4Channels as CFR_PersonID
from models import BaselineNet




model = BaselineNet(num_classes=3)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

checkpoint_path = "../models/PI.pt" 
model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
model.eval()
loss_fn = torch.nn.CrossEntropyLoss()
cumtest_loss = 0
ntest_correct = 0
ntest = 0

all_preds = []
all_targets = []
campagne_test = [
    "S2a",         # P0: ZERO-SHOT 1 
    "S6a",         # P0: ZERO-SHOT 2 
    "S4b",         # P0: STRESS TEST 
    "S3a", "S5a",  # P1: 
    "S7a"          # P2: 
]

attivita_target = ["W","R"]
batch_size = 32
num_workers = 0
pin_memory = torch.cuda.is_available()

test_dataset  = CFR_PersonID(folder="../data/doppler_traces/", campaigns=campagne_test, target_activities=attivita_target, split_mode="test")

test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)


print("\n--- Model evaluation ---")
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


class_f1_scores = f1_score(all_targets, all_preds, average=None)

test_conf = np.exp(-test_loss) * 100

print(f"Test Loss: {test_loss:.4f}")
print(f"Final Accuracy: {test_acc*100:.2f}% | Final macro F1: {test_macro_f1*100:.2f}%\n")
print(f"F1-Score for each class -> P0: {class_f1_scores[0]*100:.2f}% | P1: {class_f1_scores[1]*100:.2f}% | P2: {class_f1_scores[2]*100:.2f}%")


cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2], normalize='true')
fig, ax = plt.subplots(figsize=(9, 7))


display_labels = [
    f'P0\n[F1: {class_f1_scores[0]*100:.1f}%]', 
    f'P1\n[F1: {class_f1_scores[1]*100:.1f}%]', 
    f'P2\n[F1: {class_f1_scores[2]*100:.1f}%]'
]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(cmap='Blues', values_format='.1%', ax=ax)

plt.title(f"Confusion Matrix (Normalized)\nAcc: {test_acc*100:.1f}% | Macro F1: {test_macro_f1*100:.1f}%", pad=15)
plt.xlabel("predicted label", labelpad=10)
plt.ylabel("true label", labelpad=10)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()