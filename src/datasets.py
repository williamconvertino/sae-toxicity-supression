import csv
import random
import requests
from pathlib import Path
from torch.utils.data import Dataset
from datasets import load_dataset

EXPERIMENT_DATASET_DICT = {
    # "paradetox": lambda: load_paradetox(), # Dataset does not perform very well
    
    # "real_toxicity_0.5": lambda: load_real_toxicity(toxicity_threshold=0.5),
    # "real_toxicity_0.7": lambda: load_real_toxicity(toxicity_threshold=0.7),
    # "real_toxicity_0.9": lambda: load_real_toxicity(toxicity_threshold=0.9),
    
    # "nqopen_best": lambda: load_nqopen(only_best=True),
    "nqopen_all": lambda: load_nqopen(only_best=False),
}

HF_CACHE_DIR = Path("../cached")
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

class LabeledTextDataset(Dataset):

    def __init__(self, texts, labels):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

def split_dataset_stratified(
    dataset: LabeledTextDataset,
    train_frac=TRAIN_FRAC,
    val_frac=VAL_FRAC,
    test_frac=TEST_FRAC,
):
 
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-9, \
        "Train/val/test fractions must sum to 1."

    # Separate indices per class
    pos_idx = [i for i, y in enumerate(dataset.labels) if y == 1]
    neg_idx = [i for i, y in enumerate(dataset.labels) if y == 0]

    random.shuffle(pos_idx)
    random.shuffle(neg_idx)

    def split_indices(idxs):
        n = len(idxs)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_i = idxs[:n_train]
        val_i = idxs[n_train:n_train + n_val]
        test_i = idxs[n_train + n_val:]
        return train_i, val_i, test_i

    pos_train, pos_val, pos_test = split_indices(pos_idx)
    neg_train, neg_val, neg_test = split_indices(neg_idx)

    train_idx = pos_train + neg_train
    val_idx = pos_val + neg_val
    test_idx = pos_test + neg_test

    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)

    def subset(indices):
        return LabeledTextDataset(
            [dataset.texts[i] for i in indices],
            [dataset.labels[i] for i in indices],
        )

    return {
        "train": subset(train_idx),
        "val": subset(val_idx),
        "test": subset(test_idx),
    }

def load_paradetox():

    data_dir = Path("../datasets")
    data_dir.mkdir(exist_ok=True, parents=True)

    path = data_dir / "paradetox.tsv"

    if not path.exists():
        print("Downloading paradetox.tsv...")
        response = requests.get(
            "https://raw.githubusercontent.com/s-nlp/paradetox/main/paradetox/paradetox.tsv"
        )
        response.raise_for_status()
        path.write_bytes(response.content)
        print(f"Saved paradetox.tsv to: {path.resolve()}")

    texts = []
    labels = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            toxic_text = row["toxic"]
            if toxic_text and toxic_text.strip():
                texts.append(toxic_text.strip())
                labels.append(1)

            for key in ("neutral1", "neutral2", "neutral3"):
                text = row.get(key, "")
                if text and text.strip():
                    texts.append(text.strip())
                    labels.append(0)

    dataset = LabeledTextDataset(texts, labels)
    return split_dataset_stratified(dataset)


def load_nqopen(only_best: bool):

    print("Loading NQOpen from HF (with caching)...")

    ds = load_dataset(
        "baonn/nqopen",
        "generation",
        split="validation",
        cache_dir=str(HF_CACHE_DIR),
    )

    texts = []
    labels = []

    def format_qa(q, a):
        return f"Question: {q}\nAnswer: {a}"

    for row in ds:
        q = row["question"]

        inc = row["incorrect_answers"]
        if inc and inc.strip():
            texts.append(format_qa(q, inc.strip()))
            labels.append(1)

        if only_best:
            best = row["best_answer"]
            if best and best.strip():
                texts.append(format_qa(q, best.strip()))
                labels.append(0)
        else:
            all_correct = row["correct_answers"]
            if all_correct:
                for ans in all_correct.split(";"):
                    ans = ans.strip()
                    if ans:
                        texts.append(format_qa(q, ans))
                        labels.append(0)

    dataset = LabeledTextDataset(texts, labels)
    return split_dataset_stratified(dataset)


def load_real_toxicity(
    toxicity_threshold: float = 0.5,
    max_size: int = 10_000,
):

    print("Loading RealToxicityPrompts...")

    rtp = load_dataset(
        "allenai/real-toxicity-prompts",
        split="train",
        cache_dir=str(HF_CACHE_DIR),
    )

    positive_texts = []

    def extract_text(field):
        if isinstance(field, dict) and "text" in field:
            return field["text"]
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
            if p_text and c_text:
                positive_texts.append(f"{p_text} {c_text}")

    positive_texts = list(dict.fromkeys(positive_texts))
    if len(positive_texts) > max_size:
        positive_texts = positive_texts[:max_size]

    print(f"Toxic samples ≥ {toxicity_threshold}: {len(positive_texts)}")

    print("Loading Anthropic harmless-base...")

    harmless = load_dataset(
        "Anthropic/hh-rlhf",
        data_files={"train": "harmless-base/train.jsonl.gz"},
        split="train",
        cache_dir=str(HF_CACHE_DIR),
    )

    negative_texts = []
    for row in harmless:
        chosen = row.get("chosen", "")
        if chosen:
            negative_texts.append(chosen)

    negative_texts = list(dict.fromkeys(negative_texts))
    if len(negative_texts) > max_size:
        negative_texts = negative_texts[:max_size]

    print(f"Harmless samples (cap={max_size}): {len(negative_texts)}")

    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)

    dataset = LabeledTextDataset(texts, labels)
    return split_dataset_stratified(dataset)