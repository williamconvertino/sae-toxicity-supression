import csv
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datasets import load_dataset

def load_model():

    sae = SAE.from_pretrained(
        release="gemma-2b-it-res-jb",
        sae_id="blocks.12.hook_resid_post",
        device="cuda"
    )

    model = HookedTransformer.from_pretrained(
        "gemma-2b-it",
        device="cuda",
        dtype="float32"
    )

    return model, sae

def decode_feature_tokens(sae, model, feature_idx, k=10):
    w_dec = sae.W_dec[feature_idx]
    W_U = model.W_U
    scores = w_dec @ W_U
    topk = scores.topk(k)

    tokens = [model.tokenizer.decode([tid.item()]) for tid in topk.indices]
    values = topk.values.tolist()

    return list(zip(tokens, values))

class TextDataset(Dataset):

    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def load_paradetox():

    path = Path("paradetox.tsv")
    toxic = []
    neutral = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            toxic.append(row["toxic"])
            
            for key in ("neutral1", "neutral2", "neutral3"):
                text = row.get(key, "")
                if text and text.strip():
                    neutral.append(text.strip())

    return TextDataset(toxic), TextDataset(neutral)


def load_nqopen(only_best: bool):

    print("Loading NQOpen from HF...")
    ds = load_dataset("baonn/nqopen", "generation", split="validation")

    positive = []
    negative = []

    def format_qa(question, answer):
        return f"Question: {question}\nAnswer: {answer}"

    for row in ds:
        q = row["question"]
        
        # POSITIVE dataset: incorrect answers
        inc = row["incorrect_answers"]
        if inc and inc.strip():
            positive.append(format_qa(q, inc.strip()))

        # NEGATIVE dataset: either best answer or all correct answers
        if only_best:
            best = row["best_answer"]
            if best and best.strip():
                negative.append(format_qa(q, best.strip()))
        else:

            all_correct = row["correct_answers"]
            if all_correct:
                for ans in all_correct.split(";"):
                    ans = ans.strip()
                    if ans:
                        negative.append(format_qa(q, ans))

    return TextDataset(positive), TextDataset(negative)


def load_real_toxicity(
    toxicity_threshold: float = 0.5,
    max_size: int = 10_000,
):

    print("Loading RealToxicityPrompts...")
    rtp = load_dataset("allenai/real-toxicity-prompts", split="train")

    positive = []

    def extract_text(json_field):
        if isinstance(json_field, dict) and "text" in json_field:
            return json_field["text"]
        return None

    for row in rtp:
        p = row["prompt"]
        c = row["continuation"]

        p_text = extract_text(p)
        c_text = extract_text(c)

        toxic_val = None
        if isinstance(c, dict) and "toxicity" in c:
            toxic_val = c["toxicity"]
        elif isinstance(p, dict) and "toxicity" in p:
            toxic_val = p["toxicity"]

        if toxic_val is not None and toxic_val >= toxicity_threshold:
            if p_text:
                positive.append(p_text)
            if c_text:
                positive.append(c_text)

    positive = list(dict.fromkeys(positive))

    if len(positive) > max_size:
        positive = positive[:max_size]

    print(f"Positive samples (toxic ≥ {toxicity_threshold}): {len(positive)}")

    print("Loading Anthropic harmless...")
    neg = load_dataset(
        "Anthropic/hh-rlhf",
        data_files={"train": "harmless-base/train.jsonl.gz"},
        split="train"
    )

    negative = []
    for row in neg:
        inp = row.get("chosen", "")
        if inp:
            negative.append(inp)
        
    negative = list(dict.fromkeys(negative))

    if len(negative) > max_size:
        negative = negative[:max_size]

    print(f"Negative samples (harmless-base cap={max_size}): {len(negative)}")

    if len(positive) < 100:
        raise ValueError(
            f"RealToxicity positive dataset too small ({len(positive)}). "
            f"Try lowering toxicity_threshold."
        )

    if len(negative) < 100:
        raise ValueError(
            f"Harmless-base negative dataset too small ({len(negative)}). "
            f"This should not happen unless max_size is very small."
        )

    return TextDataset(positive), TextDataset(negative)
