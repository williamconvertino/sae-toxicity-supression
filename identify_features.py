import os
import csv
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from util import load_model, decode_feature_tokens, load_paradetox, load_nqopen, load_real_toxicity

DEVICE = torch.device("cuda")

def compute_mean_sae_activation(model, sae, dataset, batch_size=8, max_length=64):

    tok = model.tokenizer
    d_sae = sae.cfg.d_sae
    running_sum = torch.zeros(d_sae, device="cpu")
    running_count = 0

    n = len(dataset)
    n_batches = (n + batch_size - 1) // batch_size

    for i in tqdm(range(0, n, batch_size), total=n_batches, desc="Streaming activations"):
        batch_texts = [dataset[j] for j in range(i, min(i + batch_size, n))]

        toks = tok.batch_encode_plus(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )["input_ids"].to(DEVICE)

        _, cache = model.run_with_cache(toks, prepend_bos=True)

        acts = cache[sae.cfg.metadata.hook_name]
        feats = sae.encode(acts).detach().cpu()

        running_sum += feats.sum(dim=(0, 1))
        running_count += feats.shape[0] * feats.shape[1]

        del cache, toks, acts, feats
        torch.cuda.empty_cache()

    return running_sum / running_count

def analyze_features(analysis_name, model, sae, positive_dataset, negative_dataset, k=20):

    save_dir = Path(analysis_name)
    save_dir.mkdir(exist_ok=True, parents=True)

    pos_path = save_dir / "positive.pt"
    neg_path = save_dir / "negative.pt"

    if pos_path.exists():
        print(f"Loading cached mean positive activations from {pos_path}")
        pos_mean = torch.load(pos_path)
    else:
        print("Computing positive mean activations…")
        pos_mean = compute_mean_sae_activation(model, sae, positive_dataset)
        torch.save(pos_mean, pos_path)

    if neg_path.exists():
        print(f"Loading cached mean negative activations from {neg_path}")
        neg_mean = torch.load(neg_path)
    else:
        print("Computing negative mean activations…")
        neg_mean = compute_mean_sae_activation(model, sae, negative_dataset)
        torch.save(neg_mean, neg_path)

    diff = pos_mean - neg_mean

    F_plus = diff.topk(k).indices         # positive-aligned
    F_minus = (-diff).topk(k).indices     # negative-aligned (suppressing)

    plt.figure(figsize=(10, 5))
    plt.title(f"Mean SAE Feature Activation Difference ({analysis_name})")
    plt.plot(diff.cpu().numpy())
    plt.xlabel("Feature Index")
    plt.ylabel("Δ Activation")
    plt.axhline(0, color="black", linewidth=1)
    plt.tight_layout()
    plt.savefig(save_dir / "activation_diff.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(range(k), diff[F_plus].cpu().numpy())
    plt.xticks(range(k), [f"{i}" for i in F_plus.tolist()], rotation=45)
    plt.title("Top Positive-Aligned Features (F⁺)")
    plt.ylabel("Δ Activation")
    plt.tight_layout()
    plt.savefig(save_dir / "top_positive_features.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(range(k), diff[F_minus].cpu().numpy())
    plt.xticks(range(k), [f"{i}" for i in F_minus.tolist()], rotation=45)
    plt.title("Top Negative-Aligned Features (F⁻)")
    plt.ylabel("Δ Activation")
    plt.tight_layout()
    plt.savefig(save_dir / "top_negative_features.png", dpi=200)
    plt.close()

    print("\n===== Sample Decodings of Top Positive-Aligned Features =====")
    for f in F_plus[:5]:
        decoded = decode_feature_tokens(sae, model, f.item(), k=10)
        print(f"\nFeature {f.item()}:")
        for tok, score in decoded:
            print(f"  {tok!r}\t{score:.4f}")


def main():

    print("\n=== Running paradetox ===")
    model, sae = load_model()
    toxic_dataset, neutral_dataset = load_paradetox()
    print(f"Loaded {len(toxic_dataset)} toxic and {len(neutral_dataset)} neutral examples.")

    analyze_features(
        analysis_name="paradetox",
        model=model,
        sae=sae,
        positive_dataset=toxic_dataset,
        negative_dataset=neutral_dataset,
    )

    print("\n=== Running NQOpen (only best answer) ===")
    nq_pos, nq_neg = load_nqopen(only_best=True)
    print(f"NQOpen only_best: {len(nq_pos)} positive, {len(nq_neg)} negative")

    analyze_features(
        analysis_name="nqopen_onlybest",
        model=model,
        sae=sae,
        positive_dataset=nq_pos,
        negative_dataset=nq_neg,
    )

    print("\n=== Running NQOpen (all correct answers) ===")
    nq_pos2, nq_neg2 = load_nqopen(only_best=False)
    print(f"NQOpen all_correct: {len(nq_pos2)} positive, {len(nq_neg2)} negative")

    analyze_features(
        analysis_name="nqopen_allneg",
        model=model,
        sae=sae,
        positive_dataset=nq_pos2,
        negative_dataset=nq_neg2,
    )

    print("\n=== Running RealToxicity analysis ===")

    pos_rt, neg_rt = load_real_toxicity(
        toxicity_threshold=0.8,
        max_size=20_000,
    )

    analyze_features(
        analysis_name="real_toxicity",
        model=model,
        sae=sae,
        positive_dataset=pos_rt,
        negative_dataset=neg_rt,
    )

if __name__ == "__main__":
    main()
