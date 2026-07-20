import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay

def evaluate_model(model, test_loader, device, label_map, save_dir="./plot_data"):

    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_preds = []
    all_targets = []

    # define the different class names based on the label_map specified in the dataset file
    id_to_label = {}
    for label, id in label_map.items():
        if id_to_label.get(id) is None:
            id_to_label[id] = [label]
        else: id_to_label[id].append(label)
    class_names = ["-".join(id_to_label[i]) for i in range(len(id_to_label))]

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Metrics of Evaluation
    overall_acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')

    print("Model Evaluation Results:")
    print(f"Accuracy : {overall_acc * 100:.2f}%")
    print(f"Macro F1-Score  : {macro_f1 * 100:.2f}%")


    # Confusion Matrix with absolute counts
    cm_abs = confusion_matrix(all_targets, all_preds)
    
    # Confusion Matrix normalized for each row -> recall
    cm_norm = confusion_matrix(all_targets, all_preds, normalize='true')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    disp_abs = ConfusionMatrixDisplay(confusion_matrix=cm_abs, display_labels=class_names)
    disp_abs.plot(ax=axes[0], cmap='Blues', colorbar=False, values_format='d')
    axes[0].set_title("Absolute Confusion Matrix", fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)

    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp_norm.plot(ax=axes[1], cmap='Greens', colorbar=False, values_format='.1%')
    axes[1].set_title("Normalized Confusion Matrix (Recall)", fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    return {
        'accuracy': overall_acc,
        'macro_f1': macro_f1,
        'confusion_matrix': cm_abs,
        'normalized_confusion_matrix': cm_norm
    }

# Esempio di utilizzo nello script principale:
# evaluate_model(model, test_loader, device, train_dataset.LABEL_MAP)