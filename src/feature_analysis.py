import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import textwrap

DEVICE = torch.device("cuda")

def compute_mean_sae_activations(
    model,
    sae,
    dataset,
    batch_size=8,
    max_length=64,
):

    tokenizer = model.tokenizer
    d_sae = sae.cfg.d_sae

    running_sum = torch.zeros(d_sae, device="cpu")
    running_count = 0

    texts = [ex["text"] for ex in dataset]

    n = len(texts)
    n_batches = (n + batch_size - 1) // batch_size

    for i in tqdm(range(0, n, batch_size), total=n_batches, desc="Streaming activations"):
        batch_texts = texts[i : i + batch_size]

        toks = tokenizer.batch_encode_plus(
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

def load_or_generate_activations(
    experiment_name,
    model,
    sae,
    dataset_dict,
    batch_size=8,
    max_length=64,
):

    save_dir = Path("./activations") / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    pos_path = save_dir / "positive.pt"
    neg_path = save_dir / "negative.pt"

    train_data = dataset_dict["train"]

    positive = [x for x in train_data if x["label"] == 1]
    negative = [x for x in train_data if x["label"] == 0]

    # Positive
    if pos_path.exists():
        print(f"[{experiment_name}] Loading cached positive activations: {pos_path}")
        pos_mean = torch.load(pos_path)
    else:
        print(f"[{experiment_name}] Computing positive activations…")
        pos_mean = compute_mean_sae_activations(model, sae, positive, batch_size, max_length)
        torch.save(pos_mean, pos_path)

    # Negative
    if neg_path.exists():
        print(f"[{experiment_name}] Loading cached negative activations: {neg_path}")
        neg_mean = torch.load(neg_path)
    else:
        print(f"[{experiment_name}] Computing negative activations…")
        neg_mean = compute_mean_sae_activations(model, sae, negative, batch_size, max_length)
        torch.save(neg_mean, neg_path)

    return pos_mean, neg_mean

def analyze_features(
    experiment_name,
    pos_mean,
    neg_mean,
    k=20
):
    
    out_dir = Path("./figures/feature_analysis") / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    diff = pos_mean - neg_mean
    F_plus = diff.topk(k).indices
    F_minus = (-diff).topk(k).indices

    # ===== Main difference plot =====
    plt.figure(figsize=(10, 5))
    plt.title(f"Mean SAE Feature Activation Difference ({experiment_name})")
    plt.plot(diff.cpu().numpy())
    plt.xlabel("Feature Index")
    plt.ylabel("Δ Activation (positive - negative)")
    plt.axhline(0, color="black", linewidth=1)
    plt.tight_layout()
    plt.savefig(out_dir / "activation_diff.png", dpi=200)
    plt.close()

    # ===== Top positive features =====
    plt.figure(figsize=(8, 4))
    plt.bar(range(k), diff[F_plus].cpu().numpy())
    plt.xticks(range(k), [str(i.item()) for i in F_plus], rotation=45)
    plt.title("Top Positive-Aligned Features (F⁺)")
    plt.ylabel("Δ Activation")
    plt.tight_layout()
    plt.savefig(out_dir / "top_positive_features.png", dpi=200)
    plt.close()

    # ===== Top negative features =====
    plt.figure(figsize=(8, 4))
    plt.bar(range(k), diff[F_minus].cpu().numpy())
    plt.xticks(range(k), [str(i.item()) for i in F_minus], rotation=45)
    plt.title("Top Negative-Aligned Features (F⁻)")
    plt.ylabel("Δ Activation")
    plt.tight_layout()
    plt.savefig(out_dir / "top_negative_features.png", dpi=200)
    plt.close()

    return {
        "diff": diff,
        "F_plus": F_plus,
        "F_minus": F_minus,
    }

def decode_feature_tokens(sae, model, feature_idx, k=20):

    w_dec = sae.W_dec[feature_idx]
    W_U = model.W_U
    scores = w_dec @ W_U
    topk = scores.topk(k)

    tokens = [model.tokenizer.decode([tid.item()]) for tid in topk.indices]
    values = topk.values.tolist()

    return list(zip(tokens, values))


def compute_prompt_feature_activations(
    model,
    sae,
    dataset,
    feature_indices,
    batch_size=8,
    max_length=64,
):

    tokenizer = model.tokenizer
    texts = [ex["text"] for ex in dataset]
    n = len(texts)
    num_features = len(feature_indices)

    all_activations = torch.zeros(n, num_features, dtype=torch.float32)
    feature_indices = torch.tensor(feature_indices, dtype=torch.long, device=DEVICE)

    n_batches = (n + batch_size - 1) // batch_size

    for start in tqdm(
        range(0, n, batch_size),
        total=n_batches,
        desc="Streaming prompt-feature activations",
    ):
        end = min(start + batch_size, n)
        batch_texts = texts[start:end]

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

        tracked = feats[:, :, feature_indices]
        prompt_acts = tracked.mean(dim=1).detach().cpu()

        all_activations[start:end] = prompt_acts

        del cache, toks, acts, feats, tracked, prompt_acts
        torch.cuda.empty_cache()

    return all_activations, texts


def analyze_feature_tokens_and_prompts(
    experiment_name,
    model,
    sae,
    dataset_dict,
    diff,
    F_plus,
    F_minus,
    n_features_for_text=10,
    k_tokens=20,
    k_prompts=20,
    batch_size=8,
    max_length=64,
):
    
    out_dir = Path("./figures/feature_analysis") / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_tokens_path = out_dir / "top_tokens_positive.txt"
    neg_tokens_path = out_dir / "top_tokens_negative.txt"
    pos_prompts_path = out_dir / "top_prompts_positive.txt"
    neg_prompts_path = out_dir / "top_prompts_negative.txt"

    need_tokens = (not pos_tokens_path.exists()) or (not neg_tokens_path.exists())
    need_prompts = (not pos_prompts_path.exists()) or (not neg_prompts_path.exists())

    if not need_tokens and not need_prompts:
        print(f"[{experiment_name}] Token & prompt analyses already exist; skipping.")
        return

    F_plus_top = F_plus[:n_features_for_text]
    F_minus_top = F_minus[:n_features_for_text]

    # Token analysis
    if need_tokens:
        print(f"[{experiment_name}] Computing top tokens for F⁺/F⁻ features…")

        # Positive features
        with pos_tokens_path.open("w", encoding="utf-8") as f:
            f.write("===== Top Tokens for Positive-Aligned Features (F⁺) =====\n\n")
            for idx in F_plus_top:
                idx_int = idx.item()
                delta = diff[idx_int].item()
                f.write(f"Feature {idx_int} (Δ = {delta:.4f})\n")
                f.write("-" * 60 + "\n")
                token_scores = decode_feature_tokens(sae, model, idx_int, k=k_tokens)
                for rank, (tok, score) in enumerate(token_scores, start=1):
                    tok_clean = tok.replace("\n", "\\n")
                    f.write(f"{rank:2d}. {repr(tok_clean):<30}  score = {score:.4f}\n")
                f.write("\n")

        # Negative features
        with neg_tokens_path.open("w", encoding="utf-8") as f:
            f.write("===== Top Tokens for Negative-Aligned Features (F⁻) =====\n\n")
            for idx in F_minus_top:
                idx_int = idx.item()
                delta = diff[idx_int].item()
                f.write(f"Feature {idx_int} (Δ = {delta:.4f})\n")
                f.write("-" * 60 + "\n")
                token_scores = decode_feature_tokens(sae, model, idx_int, k=k_tokens)
                for rank, (tok, score) in enumerate(token_scores, start=1):
                    tok_clean = tok.replace("\n", "\\n")
                    f.write(f"{rank:2d}. {repr(tok_clean):<30}  score = {score:.4f}\n")
                f.write("\n")

        print(f"[{experiment_name}] Saved token analysis to:")
        print(f"  {pos_tokens_path}")
        print(f"  {neg_tokens_path}")

    # Prompt Analysis
    if need_prompts:
        print(
            f"[{experiment_name}] Computing top training prompts for F⁺/F⁻ features…"
        )

        train_data = dataset_dict["train"]
        tracked_features = torch.cat([F_plus_top, F_minus_top], dim=0).tolist()

        all_acts, texts = compute_prompt_feature_activations(
            model=model,
            sae=sae,
            dataset=train_data,
            feature_indices=tracked_features,
            batch_size=batch_size,
            max_length=max_length,
        )

        num_pos = len(F_plus_top)

        wrapper = textwrap.TextWrapper(width=100, subsequent_indent=" " * 4)

        with pos_prompts_path.open("w", encoding="utf-8") as f:
            f.write("===== Top Training Prompts for Positive-Aligned Features (F⁺) =====\n\n")
            for feat_idx, feat in enumerate(F_plus_top):
                feat_int = feat.item()
                delta = diff[feat_int].item()
                f.write(f"Feature {feat_int} (Δ = {delta:.4f})\n")
                f.write("-" * 80 + "\n")

                feat_acts = all_acts[:, feat_idx]  # [num_prompts]
                top_vals, top_indices = torch.topk(feat_acts, k=min(k_prompts, len(texts)))

                for rank, (val, prompt_idx) in enumerate(
                    zip(top_vals.tolist(), top_indices.tolist()), start=1
                ):
                    prompt_text = texts[prompt_idx].replace("\n", " ")
                    wrapped = wrapper.fill(prompt_text)
                    f.write(f"{rank:2d}. activation = {val:.4f}\n")
                    f.write(f"    {wrapped}\n\n")

                f.write("\n")

        with neg_prompts_path.open("w", encoding="utf-8") as f:
            f.write("===== Top Training Prompts for Negative-Aligned Features (F⁻) =====\n\n")
            for j, feat in enumerate(F_minus_top):
                feat_int = feat.item()
                delta = diff[feat_int].item()
                f.write(f"Feature {feat_int} (Δ = {delta:.4f})\n")
                f.write("-" * 80 + "\n")

                feat_acts = all_acts[:, num_pos + j]
                top_vals, top_indices = torch.topk(feat_acts, k=min(k_prompts, len(texts)))

                for rank, (val, prompt_idx) in enumerate(
                    zip(top_vals.tolist(), top_indices.tolist()), start=1
                ):
                    prompt_text = texts[prompt_idx].replace("\n", " ")
                    wrapped = wrapper.fill(prompt_text)
                    f.write(f"{rank:2d}. activation = {val:.4f}\n")
                    f.write(f"    {wrapped}\n\n")

                f.write("\n")

        print(f"[${experiment_name}] Saved prompt analysis to:")
        print(f"  {pos_prompts_path}")
        print(f"  {neg_prompts_path}")

def run_feature_analysis(
    experiment_name,
    dataset_dict,
    model,
    sae,
    batch_size=8,
    max_length=64,
    k=20,
):

    print(f"\n=== Running feature analysis: {experiment_name} ===")
    print(
        f"Dataset sizes: "
        f"{len(dataset_dict['train'])} train, "
        f"{len(dataset_dict['val'])} val, "
        f"{len(dataset_dict['test'])} test"
    )

    pos_mean, neg_mean = load_or_generate_activations(
        experiment_name,
        model,
        sae,
        dataset_dict,
        batch_size=batch_size,
        max_length=max_length,
    )

    base_results = analyze_features(
        experiment_name,
        pos_mean,
        neg_mean,
        k=k,
    )

    diff = base_results["diff"]
    F_plus = base_results["F_plus"]
    F_minus = base_results["F_minus"]

    analyze_feature_tokens_and_prompts(
        experiment_name=experiment_name,
        model=model,
        sae=sae,
        dataset_dict=dataset_dict,
        diff=diff,
        F_plus=F_plus,
        F_minus=F_minus,
        n_features_for_text=10,
        k_tokens=20,
        k_prompts=20,
        batch_size=batch_size,
        max_length=max_length,
    )

    return {
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "diff": diff,
        "F_plus": F_plus,
        "F_minus": F_minus,
    }
