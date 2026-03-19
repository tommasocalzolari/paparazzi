#!/usr/bin/env python3
"""
Train and evaluate a CNN for tree / no-tree patch classification
using a CSV file (e.g., balanced_tree_patch_labels.csv).

Expected CSV:
- One column with image paths
- One column with labels (0/1 or text like tree/no_tree)

Configuration is set at the top of the main() function.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------
# Utilities
# ---------------------------
def find_image_col(df):
    candidates = [
        "image_path", "img_path", "path", "filepath", "file_path",
        "filename", "file", "patch_path", "patch", "image"
    ]
    lower_cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_cols:
            return lower_cols[c]

    # Heuristic: string column with common image extensions
    for col in df.columns:
        if df[col].dtype == object:
            sample_vals = df[col].dropna().astype(str).head(200).str.lower()
            if sample_vals.str.contains(r"\.(jpg|jpeg|png|bmp|tif|tiff)$", regex=True).mean() > 0.3:
                return col

    raise ValueError("Could not automatically detect image path column. Set image_col in main() configuration.")


def find_label_col(df, image_col):
    candidates = ["label", "target", "class", "has_tree", "tree_present", "tree", "y"]
    lower_cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_cols and lower_cols[c] != image_col:
            return lower_cols[c]

    # Heuristic: non-image column with small number of unique values
    for col in df.columns:
        if col == image_col:
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique <= 10:
            return col

    raise ValueError("Could not automatically detect label column. Set label_col in main() configuration.")


def normalize_binary_labels(series):
    """
    Convert labels to binary 0/1.
    Supports numeric (0/1) or strings (tree/no_tree etc.).
    """
    s = series.copy()

    # Numeric case
    if np.issubdtype(s.dropna().dtype, np.number):
        s = s.astype(float)
        unique_vals = sorted(s.dropna().unique().tolist())
        if set(unique_vals).issubset({0.0, 1.0}):
            return s.astype(int)
        # fallback: positive if > 0
        return (s > 0).astype(int)

    # String case
    s = s.astype(str).str.strip().str.lower()

    pos_tokens = {"1", "true", "t", "yes", "y", "tree", "present", "has_tree"}
    neg_tokens = {"0", "false", "f", "no", "n", "no_tree", "non_tree", "not_tree", "background", "none", "absent"}

    unique_vals = set(s.dropna().unique().tolist())

    # Direct token mapping if possible
    if unique_vals.issubset(pos_tokens.union(neg_tokens)):
        return s.map(lambda x: 1 if x in pos_tokens else 0).astype(int)

    # If exactly 2 classes, infer "tree" as positive
    if len(unique_vals) == 2:
        vals = list(unique_vals)
        if any("tree" in v for v in vals):
            return s.map(lambda x: 1 if "tree" in x and "no" not in x and "non" not in x and "not" not in x else 0).astype(int)
        # fallback deterministic mapping
        mapping = {vals[0]: 0, vals[1]: 1}
        print(f"[WARN] Ambiguous string labels; using mapping: {mapping}")
        return s.map(mapping).astype(int)

    raise ValueError(
        f"Could not convert labels to binary 0/1. Unique labels found: {sorted(unique_vals)}"
    )


# ---------------------------
# Dataset
# ---------------------------
class PatchDataset(Dataset):
    def __init__(self, df, image_col, label_col, image_root="", transform=None):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(row[self.image_col])

        if self.image_root and not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root, img_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        label = float(row[self.label_col])

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


# ---------------------------
# Model
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 128x128 -> 128x128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 16x16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),# 16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 8x8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1)  # single logit for binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)


# ---------------------------
# Train / Eval
# ---------------------------
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    probs_all, labels_all = [], []

    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs_all.extend(probs.tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())

    probs_all = np.array(probs_all)
    labels_all = np.array(labels_all)
    preds_all = (probs_all >= 0.5).astype(int)

    metrics = {
        "loss": float(np.mean(losses)),
        "acc": accuracy_score(labels_all, preds_all),
        "precision": precision_score(labels_all, preds_all, zero_division=0),
        "recall": recall_score(labels_all, preds_all, zero_division=0),
        "f1": f1_score(labels_all, preds_all, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(labels_all, probs_all)
    except Exception:
        metrics["roc_auc"] = float("nan")

    return metrics, probs_all, labels_all


def main():
    # Configuration
    csv_path = "balanced_tree_patch_labels.csv"
    image_root = ""
    image_col = "frame_path"
    label_col = "tree_present"
    img_size = 128
    batch_size = 32
    epochs = 25
    lr = 1e-3
    patience = 5
    num_workers = 2
    seed = 42
    use_cpu = False
    model_out = "tree_cnn_best.pt"
    
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load CSV
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty.")

    # Determine columns
    image_col = image_col if image_col else find_image_col(df)
    label_col = label_col if label_col else find_label_col(df, image_col)

    print(f"[INFO] Image column: {image_col}")
    print(f"[INFO] Label column: {label_col}")

    # Clean and normalize labels
    df = df[[image_col, label_col]].dropna().copy()
    df[label_col] = normalize_binary_labels(df[label_col])

    # Split data: train/val/test = 70/15/15 (stratified)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df[label_col]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df[label_col]
    )

    print(f"[INFO] Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")
    print("[INFO] Class balance (train):")
    print(train_df[label_col].value_counts(normalize=True).sort_index())

    # Transforms
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Datasets / loaders
    train_ds = PatchDataset(train_df, image_col, label_col, image_root, train_tfms)
    val_ds = PatchDataset(val_df, image_col, label_col, image_root, eval_tfms)
    test_ds = PatchDataset(test_df, image_col, label_col, image_root, eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
    model = SimpleCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    print(f"[INFO] Using device: {device}")
    print("[INFO] Starting training...")

    for epoch in range(1, epochs + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics, _, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_metrics['loss']:.4f} Acc {train_metrics['acc']:.4f} F1 {train_metrics['f1']:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} Acc {val_metrics['acc']:.4f} F1 {val_metrics['f1']:.4f} AUC {val_metrics['roc_auc']:.4f}"
        )

        # Early stopping on validation F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping triggered (patience={patience}).")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    test_metrics, test_probs, test_labels = run_epoch(model, test_loader, criterion, optimizer=None, device=device, train=False)
    test_preds = (test_probs >= 0.5).astype(int)

    print("\n=== TEST METRICS ===")
    print(f"Loss      : {test_metrics['loss']:.4f}")
    print(f"Accuracy  : {test_metrics['acc']:.4f}")
    print(f"Precision : {test_metrics['precision']:.4f}")
    print(f"Recall    : {test_metrics['recall']:.4f}")
    print(f"F1        : {test_metrics['f1']:.4f}")
    print(f"ROC-AUC   : {test_metrics['roc_auc']:.4f}")

    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, digits=4))

    # Save model
    os.makedirs(os.path.dirname(model_out) if os.path.dirname(model_out) else ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_col": image_col,
            "label_col": label_col,
            "img_size": img_size,
            "threshold": 0.5,
        },
        model_out
    )
    print(f"[INFO] Saved best model to: {model_out}")


if __name__ == "__main__":
    main()