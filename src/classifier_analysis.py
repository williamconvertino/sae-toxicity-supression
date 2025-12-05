import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda")

class FeatureClassifier(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.linear = nn.Linear(d_in, 2)

    def forward(self, x):
        return self.linear(x)

def compute_batch_features(
    model,
    sae,
    batch_texts,
    feature_indices,
    max_length=64,
):
    tokenizer = model.tokenizer

    toks = tokenizer.batch_encode_plus(
        batch_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )["input_ids"].to(DEVICE)

    _, cache = model.run_with_cache(toks, prepend_bos=True)
    acts = cache[sae.cfg.metadata.hook_name]
    feats = sae.encode(acts)

    if not torch.is_tensor(feature_indices):
        feature_indices = torch.tensor(feature_indices, device=feats.device)

    feats = feats[:, :, feature_indices]
    feat_vecs = feats.mean(dim=1)

    del cache, toks, acts, feats
    torch.cuda.empty_cache()

    return feat_vecs.cpu()

def plot_training_curves(
    log_steps,
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    save_path,
):
    plt.figure(figsize=(10, 5))
    plt.plot(log_steps, train_losses, label="Train Loss")
    plt.plot(log_steps, val_losses, label="Val Loss")
    plt.legend()
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.tight_layout()
    plt.savefig(save_path / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(log_steps, train_accs, label="Train Acc")
    plt.plot(log_steps, val_accs, label="Val Acc")
    plt.legend()
    plt.xlabel("Training Step")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.tight_layout()
    plt.savefig(save_path / "accuracy_curve.png", dpi=200)
    plt.close()


def evaluate_classifier(classifier, model, sae, split_data, feature_indices, batch_size=8, max_length=64):
    classifier.eval()

    all_labels, all_preds, all_probs = [], [], []
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = len(split_data)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            idxs = range(i, min(i + batch_size, n))
            batch = [split_data[j] for j in idxs]

            batch_texts = [x["text"] for x in batch]
            batch_labels = torch.tensor([x["label"] for x in batch], device=DEVICE)

            feats = compute_batch_features(model, sae, batch_texts, feature_indices, max_length).to(DEVICE)
            logits = classifier(feats)

            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)

            all_labels.extend(batch_labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

            total_loss += loss_fn(logits, batch_labels).item()

    labels_t = torch.tensor(all_labels)
    preds_t = torch.tensor(all_preds)

    accuracy = (labels_t == preds_t).float().mean().item()
    tp = ((labels_t == 1) & (preds_t == 1)).sum().item()
    fp = ((labels_t == 0) & (preds_t == 1)).sum().item()
    fn = ((labels_t == 1) & (preds_t == 0)).sum().item()
    f1 = tp / (tp + 0.5 * (fp + fn) + 1e-8)

    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(all_labels, all_probs)
    except:
        auroc = float("nan")

    avg_loss = total_loss / max(1, (n + batch_size - 1) // batch_size)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1": f1,
        "auroc": auroc,
    }

def train_and_evaluate_classifier(
    model,
    sae,
    dataset_dict,
    feature_indices,
    save_dir,
    num_epochs=5,
    batch_size=8,
    max_length=64,
):

    save_dir.mkdir(parents=True, exist_ok=True)

    train_data = dataset_dict["train"]
    val_data   = dataset_dict["val"]
    test_data  = dataset_dict["test"]

    d_in = len(feature_indices)
    classifier = FeatureClassifier(d_in=d_in).to(DEVICE)

    opt = optim.Adam(classifier.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Logging storage
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    log_steps = []

    for epoch in range(num_epochs):
        classifier.train()
        n_train = len(train_data)
        step_interval = max(1, n_train // 10)  # log ~10 times per epoch

        for i in tqdm(range(0, n_train, batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):

            idxs = range(i, min(i + batch_size, n_train))
            batch = [train_data[j] for j in idxs]

            batch_texts = [x["text"] for x in batch]
            batch_labels = torch.tensor([x["label"] for x in batch], device=DEVICE)

            feats = compute_batch_features(model, sae, batch_texts, feature_indices, max_length).to(DEVICE)

            opt.zero_grad()
            logits = classifier(feats)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            opt.step()

            # Logging every step_interval
            global_step = epoch * n_train + i
            if (i % step_interval == 0) or i + batch_size >= n_train:
                classifier.eval()

                preds = torch.argmax(logits, dim=-1)
                train_acc = (preds == batch_labels).float().mean().item()

                val_metrics = evaluate_classifier(classifier, model, sae, val_data, feature_indices, batch_size, max_length)

                train_losses.append(loss.item())
                val_losses.append(val_metrics["loss"])
                train_accs.append(train_acc)
                val_accs.append(val_metrics["accuracy"])
                log_steps.append(global_step)

    # Save training curves
    plot_training_curves(
        log_steps,
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        save_dir,
    )

    # Save classifier
    torch.save(classifier.state_dict(), save_dir / "model.pt")

    # Final test results
    test_results = evaluate_classifier(
        classifier,
        model,
        sae,
        test_data,
        feature_indices,
        batch_size,
        max_length,
    )

    with open(save_dir / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    return test_results

def load_or_generate_classifier(
    experiment_name,
    k,
    k_type,
    feature_indices,
    model,
    sae,
    dataset_dict,
    batch_size=8,
    max_length=64,
):
    save_dir = Path(f"figures/classifier_analysis/{experiment_name}/{k_type}/top_k={k}")
    model_path = save_dir / "model.pt"
    results_path = save_dir / "test_results.json"

    if model_path.exists() and results_path.exists():
        print(f"[{experiment_name}] Cached classifier | k={k}, type={k_type}")
        with open(results_path, "r") as f:
            return json.load(f)

    print(f"[{experiment_name}] Training classifier | k={k}, type={k_type}")
    return train_and_evaluate_classifier(
        model=model,
        sae=sae,
        dataset_dict=dataset_dict,
        feature_indices=feature_indices,
        save_dir=save_dir,
        batch_size=batch_size,
        max_length=max_length,
    )


def plot_metrics_vs_k(experiment_name, k_type, results_dict):
    save_dir = Path(f"figures/classifier_analysis/{experiment_name}/{k_type}")
    save_dir.mkdir(parents=True, exist_ok=True)

    ks_numeric = [1, 5, 10]
    ks_all = ["all"]

    metrics = ["accuracy", "f1", "auroc", "loss"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))

        y_small = [results_dict[k][metric] for k in ks_numeric]
        y_all = results_dict["all"][metric]

        plt.plot(ks_numeric, y_small, marker="o", label="Small k")

        # Fake second axis position for "all"
        plt.plot([15], [y_all], marker="s", color="red", label="k=all")

        plt.xlabel("k (with axis break before 'all')")
        plt.ylabel(metric)
        plt.title(f"{metric} vs k ({k_type})")
        plt.legend()
        plt.tight_layout()

        plt.savefig(save_dir / f"{metric}_vs_k.png", dpi=200)
        plt.close()

def run_classifier_analysis(
    experiment_name,
    dataset_dict,
    model,
    sae,
    pos_mean,
    neg_mean,
    batch_size=8,
    max_length=64,
):

    diff = pos_mean - neg_mean
    sorted_positive = diff.argsort(descending=True)
    sorted_negative = diff.argsort(descending=False)

    results_all_types = {}

    for k_type in ["positive", "negative", "both"]:
        print(f"\n====== Classifier Analysis ({k_type}) ======")

        ks = [1, 5, 10, "all"]
        results = {}

        for k in ks:
            if k == "all":
                if k_type == "positive":
                    feature_indices = sorted_positive.tolist()

                elif k_type == "negative":
                    feature_indices = sorted_negative.tolist()

                else:
                    # include all pos and all neg
                    feature_indices = torch.cat([sorted_positive, sorted_negative]).tolist()

            else:
                if k_type == "positive":
                    feature_indices = sorted_positive[:k].tolist()

                elif k_type == "negative":
                    feature_indices = sorted_negative[:k].tolist()

                else:  # both
                    feature_indices = torch.cat([
                        sorted_positive[:k],
                        sorted_negative[:k]
                    ]).tolist()

            results[k] = load_or_generate_classifier(
                experiment_name,
                k,
                k_type,
                feature_indices,
                model,
                sae,
                dataset_dict,
                batch_size,
                max_length,
            )

        plot_metrics_vs_k(experiment_name, k_type, results)

        results_all_types[k_type] = results

    return results_all_types
