import torch, json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from recurrent import ConvolutionalRecurrentNet
from dataset import CFR
from model_evaluation import evaluate_model

history_path = "plot_data/training_history_recurrent.json"
with open(history_path, "r") as f:
    history = json.load(f)

test_dataset = CFR(folder="../data/doppler_traces/S1", campaigns=["c"], split_mode="test")

batch_size = 64
num_workers = 0
pin_memory = torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
plt.savefig("./plot_data/training_curves_recurrent.png")  # Salva il grafico come immagine sul PC
plt.show()


# TESTING
checkpoint_path = "./models/recurrent_model.pt"
model = ConvolutionalRecurrentNet()
model.load_state_dict(torch.load(checkpoint_path))

model = model.to(device)
results = evaluate_model(model, test_dataloader, device, test_dataset.LABEL_MAP, save_dir="./plot_data")

