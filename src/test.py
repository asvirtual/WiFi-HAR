    from dataset2 import Normalize
    import torch, json
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from baseline4 import BaselineNet
    from dataset2 import CFR
    from model_evaluation2 import evaluate_model

    history_path = "plot_data/training_history_baseline4.json"
    with open(history_path, "r") as f:
        history = json.load(f)

    checkpoint_path = "./models/baseline4_model.pt"
    checkpoint = torch.load(checkpoint_path)

    if 'train_mean' in checkpoint and 'train_std' in checkpoint:
        mean = checkpoint['train_mean']
        std = checkpoint['train_std']
        test_dataset = CFR(folder="../data/doppler_traces/S1", campaigns=["c"], split_mode="test", transform=Normalize(mean, std))
    else:
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
    plt.savefig("./plot_data/training_curves_baseline4.png")  # Salva il grafico come immagine sul PC
    plt.show()


    # TESTING

    model = BaselineNet()
    model.load_state_dict(checkpoint['model_state_dict'])


    model = model.to(device)
    results = evaluate_model(model, test_dataloader, device, test_dataset.LABEL_MAP, save_dir="./plot_data")

