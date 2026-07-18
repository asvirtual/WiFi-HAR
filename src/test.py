import torch, json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from baseline import BaselineNet
from dataset import CFR
from tqdm import tqdm

history_path = "plot_data/training_history_baseline.json"
with open(history_path, "r") as f:
    history = json.load(f)

test_dataset = CFR(folder="../data/doppler_traces/S1", campaigns=["c"], split_mode="test")

batch_size = 64
num_workers = 0
pin_memory = torch.cuda.is_available()

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

plt.figure(figsize=(12, 5))

# Grafico delle Loss (Train vs Validation)
plt.subplot(1, 2, 1)
plt.plot(history["train"], label="Train Loss", color="blue", lw=2)
plt.plot(history["val"], label="Validation Loss", color="orange", lw=2)
plt.title("Loss Evolution")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Grafico dell'Accuratezza di Validation
plt.subplot(1, 2, 2)
plt.plot(history["acc"], label="Val Accuracy", color="green", lw=2)
plt.title("Validation Accuracy Evolution")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png")  # Salva il grafico come immagine sul PC
plt.show()

# TESTING
checkpoint_path = "baseline_model.pt"
model = BaselineNet()
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

cumtest_loss = 0
ntest_correct = 0
ntest = 0
loss_fn = CrossEntropyLoss()

with torch.no_grad():
    test_iterator = tqdm(test_dataloader)
    for batch_x, batch_y in test_iterator:
        batch_x = batch_x.to(torch.device)
        batch_y = batch_y.to(torch.device)

        y_pred = model(batch_x)
        batch_loss = loss_fn(y_pred, batch_y)

        cumtest_loss += batch_loss.item() * batch_x.size(0)
        ntest += batch_x.size(0)

        predictions = y_pred.argmax(dim=1)
        ntest_correct += (predictions == batch_y).sum().item()

        test_iterator.set_description(f"Test loss: {batch_loss.item():.5f}")

test_loss = cumtest_loss / ntest
test_acc = ntest_correct / ntest
print(f"loss: {test_loss}, accuracy: {test_acc}")


#By using as training set the first two days on the first monitor position and the third as training set we obtain:
#Test loss: 1.2525141948078449, accuracy: 0.5575328265376641
