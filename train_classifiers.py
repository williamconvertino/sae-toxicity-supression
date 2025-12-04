import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from util import (
    load_model,
    load_paradetox,
    load_nqopen,
    load_real_toxicity,
)

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class LabeledTextDataset(Dataset):
    """
    Wraps text + binary label.
    item: (text: str, label: int)
    """

    def __init__(self, texts: List[str], labels: List[int]):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], int(self.labels[idx])


def build_labeled_dataset(positive_ds: Dataset, negative_ds: Dataset) -> LabeledTextDataset:

    pos_texts = [positive_ds[i] for i in range(len(positive_ds))]
    neg_texts = [negative_ds[i] for i in range(len(negative_ds))]

    texts = pos_texts + neg_texts
    labels = [1] * len(pos_texts) + [0] * len(neg_texts)

    rng = np.random.default_rng(seed=42)
    indices = np.arange(len(texts))
    rng.shuffle(indices)

    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]

    return LabeledTextDataset(texts, labels)


def split_dataset(
    dataset: Dataset,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:

    n = len(dataset)
    indices = np.arange(n)
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(indices)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    return train_ds, val_ds, test_ds

def load_feature_order_from_means(analysis_name: str) -> torch.Tensor:

    save_dir = Path(analysis_name)
    pos_path = save_dir / "positive.pt"
    neg_path = save_dir / "negative.pt"

    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError(
            f"Expected mean activation files not found for '{analysis_name}'. "
            f"Looking for {pos_path} and {neg_path}."
        )

    pos_mean = torch.load(pos_path, map_location="cpu")
    neg_mean = torch.load(neg_path, map_location="cpu")

    diff = pos_mean - neg_mean
    
    abs_diff = diff.abs()
    order = torch.argsort(abs_diff, descending=True)
    return order 

class SAEClassifier(nn.Module):

    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: (B, in_dim)
        return self.linear(x).squeeze(-1)


def make_collate_fn(
    model,
    sae,
    feature_indices: Optional[torch.Tensor],
    max_length: int,
    device: torch.device,
):
    tok = model.tokenizer
    hook_name = sae.cfg.metadata.hook_name
    d_sae = sae.cfg.d_sae

    if feature_indices is not None:
        feature_indices = feature_indices.to(device)

    def collate(batch):
        texts, labels = zip(*batch)

        enc = tok.batch_encode_plus(
            list(texts),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        toks = enc["input_ids"].to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(toks, prepend_bos=True)
            acts = cache[hook_name]
            feats = sae.encode(acts)

            pooled = feats.mean(dim=1)

            if feature_indices is not None:
                X = pooled[:, feature_indices]
            else:
                X = pooled

        y = torch.tensor(labels, dtype=torch.float32, device=device)

        del cache, acts, feats, pooled, enc
        torch.cuda.empty_cache()

        return X, y

    return collate

def train_classifier(
    model,
    sae,
    train_ds: Dataset,
    val_ds: Dataset,
    feature_indices: Optional[torch.Tensor],
    analysis_name: str,
    dataset_name: str,
    k_label: str,
    out_root: Path,
    num_epochs: int = 5,
    batch_size: int = 32,
    max_length: int = 64,
    lr: float = 1e-3,
) -> Dict[str, List[float]]:

    model.eval()
    sae.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in sae.parameters():
        p.requires_grad_(False)

    # Determine input dimension
    if feature_indices is None:
        in_dim = sae.cfg.d_sae
    else:
        in_dim = feature_indices.numel()

    classifier = SAEClassifier(in_dim).to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    collate_train = make_collate_fn(
        model=model,
        sae=sae,
        feature_indices=feature_indices,
        max_length=max_length,
        device=DEVICE,
    )
    collate_val = make_collate_fn(
        model=model,
        sae=sae,
        feature_indices=feature_indices,
        max_length=max_length,
        device=DEVICE,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_val,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, num_epochs + 1):
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X, y in tqdm(train_loader, desc=f"[{dataset_name} | k={k_label}] Epoch {epoch} (train)"):
            optimizer.zero_grad()
            logits = classifier(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.size(0)

            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).long()
                train_correct += (preds == y.long()).sum().item()
                train_total += y.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"[{dataset_name} | k={k_label}] Epoch {epoch} (val)"):
                logits = classifier(X)
                loss = criterion(logits, y)
                val_loss += loss.item() * y.size(0)

                preds = (torch.sigmoid(logits) >= 0.5).long()
                val_correct += (preds == y.long()).sum().item()
                val_total += y.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"[{dataset_name} | k={k_label}] "
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

    # Save curves
    out_dir = out_root / dataset_name / f"k_{k_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss curve
    plt.figure()
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{dataset_name} | k={k_label} | Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=200)
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(history["train_acc"], label="Train acc")
    plt.plot(history["val_acc"], label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset_name} | k={k_label} | Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png", dpi=200)
    plt.close()

    # Final classifier weights
    torch.save(classifier.state_dict(), out_dir / "classifier.pt")

    return history


def evaluate_classifier(
    model,
    sae,
    classifier: SAEClassifier,
    test_ds: Dataset,
    feature_indices: Optional[torch.Tensor],
    max_length: int,
) -> Dict[str, float]:

    classifier.eval()

    collate_test = make_collate_fn(
        model=model,
        sae=sae,
        feature_indices=feature_indices,
        max_length=max_length,
        device=DEVICE,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_test,
    )

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Testing"):
            logits = classifier(X)
            probs = torch.sigmoid(logits)
            all_labels.append(y.cpu())
            all_scores.append(probs.cpu())

    labels = torch.cat(all_labels).numpy()
    scores = torch.cat(all_scores).numpy()
    preds = (scores >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = float("nan")

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "auroc": float(auroc),
    }


def plot_k_vs_metrics(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    ks: List[str],
    out_root: Path,
):

    metric_names = ["accuracy", "f1", "auroc"]

    for dataset_name, k_dict in all_metrics.items():
        for metric in metric_names:
            values = []
            for k_label in ks:
                if k_label in k_dict and metric in k_dict[k_label]:
                    values.append(k_dict[k_label][metric])
                else:
                    values.append(float("nan"))

            plt.figure()
            plt.plot(ks, values, marker="o")
            plt.xlabel("k (number of SAE features)")
            plt.ylabel(metric.upper())
            plt.title(f"{dataset_name}: {metric.upper()} vs k")
            plt.tight_layout()

            out_dir = out_root / dataset_name
            out_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir / f"{metric}_vs_k.png", dpi=200)
            plt.close()

def main():
    # Root directory for all classifier results
    out_root = Path("classifier_results")
    out_root.mkdir(exist_ok=True, parents=True)

    print("Loading model and SAE...")
    model, sae = load_model()
    model.to(DEVICE)
    sae.to(DEVICE)

    TASKS = {
        "paradetox": lambda: load_paradetox(),
        "nqopen_onlybest": lambda: load_nqopen(only_best=True),
        "nqopen_allneg": lambda: load_nqopen(only_best=False),
        "real_toxicity": lambda: load_real_toxicity(
            toxicity_threshold=0.5,
            max_size=10_000,
        ),
    }

    ks_raw = [1, 3, 5, "all"]
    ks_labels = ["1", "3", "5", "all"]

    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    max_length = 64
    num_epochs = 1
    batch_size = 32

    for dataset_name, loader_fn in TASKS.items():
        print(f"\n=== Preparing dataset: {dataset_name} ===")
        pos_ds, neg_ds = loader_fn()
        print(f"  Positive examples: {len(pos_ds)}")
        print(f"  Negative examples: {len(neg_ds)}")

        labeled_ds = build_labeled_dataset(pos_ds, neg_ds)
        train_ds, val_ds, test_ds = split_dataset(labeled_ds, train_frac=0.8, val_frac=0.1, seed=42)

        print(
            f"  Split sizes: "
            f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
        )

        print(f"Loading pretrained feature ranking for {dataset_name}...")
        feature_order = load_feature_order_from_means(dataset_name)  # (d_sae,)

        all_metrics[dataset_name] = {}

        for k_raw, k_label in zip(ks_raw, ks_labels):
            if k_raw == "all":
                feature_indices = None  # use all SAE features
            else:
                k_int = int(k_raw)
                feature_indices = feature_order[:k_int]

            print(f"\n--- Training classifier for {dataset_name} with k={k_label} ---")

            history = train_classifier(
                model=model,
                sae=sae,
                train_ds=train_ds,
                val_ds=val_ds,
                feature_indices=feature_indices,
                analysis_name=dataset_name,
                dataset_name=dataset_name,
                k_label=k_label,
                out_root=out_root,
                num_epochs=num_epochs,
                batch_size=batch_size,
                max_length=max_length,
                lr=1e-3,
            )

            in_dim = sae.cfg.d_sae if feature_indices is None else feature_indices.numel()
            classifier = SAEClassifier(in_dim).to(DEVICE)

            clf_path = out_root / dataset_name / f"k_{k_label}" / "classifier.pt"
            classifier.load_state_dict(torch.load(clf_path, map_location=DEVICE))

            print(f"Evaluating classifier for {dataset_name}, k={k_label} on test set...")
            test_metrics = evaluate_classifier(
                model=model,
                sae=sae,
                classifier=classifier,
                test_ds=test_ds,
                feature_indices=feature_indices,
                max_length=max_length,
            )

            print(
                f"[TEST] {dataset_name} | k={k_label}: "
                f"accuracy={test_metrics['accuracy']:.4f}, "
                f"f1={test_metrics['f1']:.4f}, "
                f"auroc={test_metrics['auroc']:.4f}"
            )

            out_dir = out_root / dataset_name / f"k_{k_label}"
            report = {
                "dataset": dataset_name,
                "k": k_label,
                "train_history": history,
                "test_metrics": test_metrics,
            }
            with open(out_dir / "report.json", "w") as f:
                json.dump(report, f, indent=2)

            all_metrics[dataset_name][k_label] = test_metrics

    print("\nGenerating k vs metric plots for each dataset...")
    plot_k_vs_metrics(all_metrics, ks_labels, out_root)

    with open(out_root / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
