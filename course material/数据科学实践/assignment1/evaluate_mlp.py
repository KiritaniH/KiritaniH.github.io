import os
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from dataset import LeavesDataset
from mlp import MLP


def denormalize(img_tensor, mean, std):
    """
    img_tensor: (3, H, W)
    """
    img = img_tensor.clone().cpu()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = torch.clamp(img, 0, 1)
    return img


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []
    all_images = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_images.extend(images.cpu())

    return np.array(all_labels), np.array(all_preds), all_images


def save_classification_report(y_true, y_pred, class_names, save_path):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Classification report saved to {save_path}")


def save_confusion_matrix(y_true, y_pred, save_path, top_k_classes=None):
    if top_k_classes is None:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(14, 12))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix (All Classes)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        mask = np.isin(y_true, top_k_classes)
        y_true_sub = y_true[mask]
        y_pred_sub = y_pred[mask]

        label_map = {old: new for new, old in enumerate(top_k_classes)}
        y_true_sub = np.array([label_map[x] for x in y_true_sub])
        y_pred_sub = np.array([label_map[x] for x in y_pred_sub if x in label_map])

        valid_idx = [i for i, p in enumerate(y_pred[mask]) if p in label_map]
        y_true_sub = np.array([label_map[y_true[mask][i]] for i in valid_idx])
        y_pred_sub = np.array([label_map[y_pred[mask][i]] for i in valid_idx])

        cm = confusion_matrix(y_true_sub, y_pred_sub, labels=list(range(len(top_k_classes))))

        plt.figure(figsize=(8, 7))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix (Selected Classes)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    print(f"Confusion matrix saved to {save_path}")


def save_misclassified_examples(images, y_true, y_pred, idx_to_class, save_path, max_show=9):
    wrong_indices = np.where(y_true != y_pred)[0]

    if len(wrong_indices) == 0:
        print("No misclassified examples found.")
        return

    wrong_indices = wrong_indices[:max_show]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    cols = 3
    rows = math.ceil(len(wrong_indices) / cols)
    plt.figure(figsize=(12, 4 * rows))

    for i, idx in enumerate(wrong_indices):
        img = denormalize(images[idx], mean, std)
        img = img.permute(1, 2, 0).numpy()

        true_name = idx_to_class[int(y_true[idx])]
        pred_name = idx_to_class[int(y_pred[idx])]

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"True: {true_name}\nPred: {pred_name}", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Misclassified examples saved to {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ===== paths =====
    root = "leafdataset"
    model_path = "best_mlp.pth"
    batch_size = 64

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cannot find model file: {model_path}")

    # ===== transforms =====
    transforms_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ===== dataset =====
    test_dataset = LeavesDataset(root, mode='test', transforms=transforms_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ===== model =====
    model = MLP(
        input_dim=64 * 64 * 3,
        hidden_dim=1024,
        num_classes=176,
        dropout=0.0
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from: {model_path}")

    # ===== collect predictions =====
    y_true, y_pred, images = collect_predictions(model, test_loader, device)

    # ===== metrics =====
    acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n===== Test Metrics =====")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Macro Precision : {macro_precision:.4f}")
    print(f"Macro Recall    : {macro_recall:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")

    # ===== save metrics to txt =====
    with open("test_metrics.txt", "w", encoding="utf-8") as f:
        f.write("===== Test Metrics =====\n")
        f.write(f"Accuracy        : {acc:.4f}\n")
        f.write(f"Macro Precision : {macro_precision:.4f}\n")
        f.write(f"Macro Recall    : {macro_recall:.4f}\n")
        f.write(f"Macro F1        : {macro_f1:.4f}\n")
    print("Metrics saved to test_metrics.txt")

    # ===== classification report =====
    class_names = [test_dataset.idx_to_class[i] for i in range(len(test_dataset.idx_to_class))]
    save_classification_report(
        y_true,
        y_pred,
        class_names,
        save_path="classification_report.txt"
    )

    # ===== confusion matrix =====
    save_confusion_matrix(
        y_true,
        y_pred,
        save_path="confusion_matrix.png",
        top_k_classes=None
    )

    # ===== misclassified examples =====
    save_misclassified_examples(
        images,
        y_true,
        y_pred,
        idx_to_class=test_dataset.idx_to_class,
        save_path="misclassified_examples.png",
        max_show=9
    )


if __name__ == "__main__":
    main()