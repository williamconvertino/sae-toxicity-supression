
import torch
from functools import partial
import re
import matplotlib.pyplot as plt
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import json
from googleapiclient import discovery
from dotenv import load_dotenv
import os
import json
import time

DEVICE = torch.device("cuda")

HF_CACHE_DIR = Path("../cached")
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()
N_SAMPLE_GENERATIONS = 5

PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")
if PERSPECTIVE_API_KEY is None:
    raise ValueError("PERSPECTIVE_API_KEY not found in .env file")

def format_gemma_prompt(question: str) -> str:
    return (
        "<start_of_turn>user\n"
        f"{question}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )



PERSPECTIVE_CLIENT = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=PERSPECTIVE_API_KEY,
    discoveryServiceUrl=(
        "https://commentanalyzer.googleapis.com/"
        "$discovery/rest?version=v1alpha1"
    ),
    static_discovery=False,
)

def batch_max_activation(model, sae, tokens, hook_name, layer, feature_idx):
    _, cache = model.run_with_cache(
        tokens,
        stop_at_layer=layer + 1,
        names_filter=[hook_name],
    )
    sae_in = cache[hook_name]                
    sae_out = sae.encode(sae_in)                 
    sae_out = sae_out.reshape(-1, sae_out.size(-1))
    return sae_out[:, feature_idx].max().item()


def steering_hook_fn(
    activations,
    hook,
    steering_vector,
    steering_strength,
    max_act
):
    return activations + steering_strength * max_act * steering_vector


def generate_with_steering_batch(
    model,
    sae,
    prompts,
    feature_indices,
    steering_strength,
    max_new_tokens=80,
    use_global_max_act=True,
):
    """
    If feature_indices is empty OR steering_strength == 0,
    we run generation WITHOUT steering (baseline).
    """
    baseline_mode = (len(feature_indices) == 0) or (steering_strength == 0.0)

    hook_name = sae.cfg.metadata.hook_name
    layer = int(re.search(r"\.(\d+)\.", hook_name).group(1))

    tokens = model.to_tokens(
        prompts,
        prepend_bos=sae.cfg.metadata.prepend_bos
    ).to(DEVICE)

    if baseline_mode:
        out_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            stop_at_eos=True,
        )
        decoded = [model.tokenizer.decode(o) for o in out_tokens]
        return decoded, 0.0   

    indices = [i for (i, s) in feature_indices]
    signs   = torch.tensor([s for (i, s) in feature_indices], device=DEVICE).float()

    W = sae.W_dec[indices].to(DEVICE)       
    W = W * signs.unsqueeze(1)            

    steering_vector = W.sum(dim=0)         
    steering_vector = steering_vector / steering_vector.norm()
    steering_vector = steering_vector.unsqueeze(0).unsqueeze(0)

    if use_global_max_act:
        max_acts = []
        for f in feature_indices:
            max_acts.append(batch_max_activation(model, sae, tokens, hook_name, layer, f))
        max_act = max(max_acts)
    else:
        raise NotImplementedError("Only global max activation supported for now.")

    hook_fn = partial(
        steering_hook_fn,
        steering_vector=steering_vector,
        steering_strength=steering_strength,
        max_act=max_act,
    )

    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        out_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            stop_at_eos=True,
            # prepend_bos=sae.cfg.metadata.prepend_bos,
        )

    decoded = [model.tokenizer.decode(o) for o in out_tokens]

    return decoded, max_act

def save_sample_generations_toxicity(
    path,
    experiment_name,
    k_type,
    k,
    alpha,
    threshold,
    prompts,
    generations,
):

    # path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Task: toxicity\n")
        f.write(f"k_type: {k_type}\n")
        f.write(f"k: {k}\n")
        f.write(f"alpha: {alpha}\n")
        f.write(f"threshold: {threshold}\n")
        f.write("\n")

        for i, (p, g) in enumerate(zip(prompts, generations)):
            f.write(f"=== SAMPLE {i+1} ===\n")
            f.write("PROMPT:\n")
            f.write(p.strip() + "\n\n")
            f.write("GENERATION:\n")
            f.write(g.strip() + "\n\n")
            f.write("-" * 80 + "\n\n")

def save_sample_generations_hallucination(
    path,
    experiment_name,
    k_type,
    k,
    alpha,
    questions,
    best_answers,
    correct_lists,
    generations,
):

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Task: hallucination (NQOpen)\n")
        f.write(f"k_type: {k_type}\n")
        f.write(f"k: {k}\n")
        f.write(f"alpha: {alpha}\n")
        f.write("\n")

        for i, (q, best, correct, gen) in enumerate(
            zip(questions, best_answers, correct_lists, generations)
        ):
            f.write(f"=== SAMPLE {i+1} ===\n")
            f.write("QUESTION:\n")
            f.write(q.strip() + "\n\n")
            f.write("BEST ANSWER:\n")
            f.write(str(best).strip() + "\n\n")
            f.write("CORRECT ANSWERS:\n")
            f.write("; ".join(correct) + "\n\n")
            f.write("GENERATION:\n")
            f.write(gen.strip() + "\n\n")
            f.write("-" * 80 + "\n\n")


def get_toxicity_score(text, max_retries=3, delay=3.0, timeout=3.0):

    analyze_request = {
        "comment": {"text": text},
        "requestedAttributes": {"TOXICITY": {}},
        "languages": ["en"],
        "doNotStore": True,
    }

    for attempt in range(max_retries):
        try:
            response = (
                PERSPECTIVE_CLIENT.comments()
                .analyze(body=analyze_request)
                .execute()
            )

            score = (
                response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            )

            time.sleep(delay)

            return float(score)

        except Exception as e:
            print(f"[Perspective API] Error on attempt {attempt+1}/{max_retries}: {e}")
            time.sleep(timeout * (attempt + 1)) 

    return 0.0

def load_real_toxicity_prompts(threshold, max_samples=1_000):

    ds = load_dataset("allenai/real-toxicity-prompts", split="train")

    prompts = []
    for row in ds:

        if len(prompts) > max_samples:
            break

        prompt = row["prompt"]
        response = row["continuation"]

        if not isinstance(prompt, dict) or not isinstance(response, dict):
            continue
        if "text" not in prompt or "text" not in response:
            continue
        if "toxicity" not in response:
            continue

        if response["toxicity"] is not None and response["toxicity"] >= threshold:
            prompts.append(format_gemma_prompt(prompt["text"]))
    assert len(prompts) > 0, "Found no valid prompts"
    return prompts

def run_toxic_analysis(
    experiment_name,
    k_type,
    k,
    alpha,
    feature_indices,
    model,
    sae,
    max_new_tokens=80,
    batch_size=8,
):
    
    try:
        threshold = float(experiment_name.split("_")[-1])
        assert threshold is not None, f"Got None threshold from {experiment_name}"
    except:
        raise ValueError(f"Experiment '{experiment_name}' is not a real_toxicity_* experiment")

    save_dir = Path(
        f"figures/steering_toxicity/{experiment_name}/{k_type}/top_k={k}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    result_path = save_dir / f"alpha={alpha}.json"
    samples_path = save_dir / f"samples_alpha={alpha}.txt"

    if result_path.exists():
        with open(result_path, "r") as f:
            avg_tox = json.load(f)["avg_toxicity"]

        if not samples_path.exists():
            samples_path.parent.mkdir(parents=True, exist_ok=True)
 
            sample_prompts = load_real_toxicity_prompts(threshold, max_samples=N_SAMPLE_GENERATIONS)
            sample_generations, _ = generate_with_steering_batch(
                model=model,
                sae=sae,
                prompts=sample_prompts,
                feature_indices=feature_indices,
                steering_strength=alpha,
                max_new_tokens=max_new_tokens,
                use_global_max_act=True,
            )
            save_sample_generations_toxicity(
                samples_path,
                experiment_name,
                k_type,
                k,
                alpha,
                threshold,
                sample_prompts,
                sample_generations,
            )

        return avg_tox

    result_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_real_toxicity_prompts(threshold)
    print(f"Loaded {len(prompts)} prompts from {experiment_name}")

    if len(prompts) == 0:
        with open(result_path, "w") as f:
            json.dump({"avg_toxicity": 0.0}, f, indent=2)
        return 0.0

    toxicity_scores = []

    for i in tqdm(range(0, len(prompts), batch_size),
                  desc=f"Toxic eval: k={k}, {k_type}, alpha={alpha}"):

        batch = prompts[i:i + batch_size]

        generations, _ = generate_with_steering_batch(
            model=model,
            sae=sae,
            prompts=batch,
            feature_indices=feature_indices,
            steering_strength=alpha,
            max_new_tokens=max_new_tokens,
            use_global_max_act=True,
        )

        for text in generations:
            toxicity_scores.append(get_toxicity_score(text))

    avg_tox = float(sum(toxicity_scores) / len(toxicity_scores))

    with open(result_path, "w") as f:
        json.dump({"avg_toxicity": avg_tox}, f, indent=2)

    if not samples_path.exists():
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        sample_prompts = prompts[:N_SAMPLE_GENERATIONS]
        sample_generations, _ = generate_with_steering_batch(
            model=model,
            sae=sae,
            prompts=sample_prompts,
            feature_indices=feature_indices,
            steering_strength=alpha,
            max_new_tokens=max_new_tokens,
            use_global_max_act=True,
        )
        save_sample_generations_toxicity(
            samples_path,
            experiment_name,
            k_type,
            k,
            alpha,
            threshold,
            sample_prompts,
            sample_generations,
        )

    return avg_tox


def _get(result_dict, k, alpha):
    if alpha == "baseline":
        return result_dict[k]["baseline"]
    else:
        return result_dict[k][alpha]
    
def _get_metric(results, k, alpha, metric):

    if alpha == "baseline":
        return results[k]["baseline"][metric]
    else:
        return results[k][alpha][metric]


def plot_toxicity_curves(experiment_name, k_type, results):

    save_dir = Path(f"figures/steering_toxicity/{experiment_name}/{k_type}")
    save_dir.mkdir(parents=True, exist_ok=True)

    ks = [1, 5, 10, "all"]
    alphas = ["baseline", 0.5, 1.0, 3.0, 6.0]

    for alpha in alphas:
        plt.figure(figsize=(8, 5))
        small_ks = [1, 5, 10]
        tox_small = [_get(results, k, alpha) for k in small_ks]

        tox_all = results["all"][alpha]

        plt.plot(small_ks, tox_small, marker="o", label="k = 1,5,10")
        plt.scatter([15], [tox_all], color="red", label="k = all")

        plt.title(f"Toxicity vs k | {experiment_name} | {k_type} | alpha={alpha}")
        plt.xlabel("k (axis break before 'all')")
        plt.ylabel("avg toxicity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"tox_vs_k_alpha={alpha}.png", dpi=200)
        plt.close()

    for k in ks:
        plt.figure(figsize=(8, 5))
        tox_vals = [results[k][alpha] for alpha in alphas]

        plt.plot(alphas, tox_vals, marker="o")
        plt.title(f"Toxicity vs alpha | {experiment_name} | {k_type} | k={k}")
        plt.xlabel("alpha")
        plt.ylabel("avg toxicity")
        plt.tight_layout()
        plt.savefig(save_dir / f"tox_vs_alpha_k={k}.png", dpi=200)
        plt.close()

def load_nqopen_questions(max_samples=4_000):

    ds = load_dataset(
        "baonn/nqopen",
        "generation",
        split="validation",
        cache_dir=str(HF_CACHE_DIR),
    )

    questions = []
    best_answers = []
    correct_lists = []

    for row in ds:
        q = row.get("question", None)
        q = format_gemma_prompt(q)
        best = row.get("best_answer", None)
        correct_str = row.get("correct_answers", "")

        if q is None or best is None:
            continue

        if isinstance(correct_str, str):
            correct_split = [c.strip() for c in correct_str.split(";") if c.strip()]
        else:
            correct_split = []

        questions.append(q)
        best_answers.append(best)
        correct_lists.append(correct_split)

        if len(questions) >= max_samples:
            break

    assert len(questions) > 0, "Found no valid NQOpen samples"
    return questions, best_answers, correct_lists

def run_hallucination_analysis(
    experiment_name,
    k_type,
    k,
    alpha,
    feature_indices,
    model,
    sae,
    max_new_tokens=80,
    batch_size=8,
):
    if not experiment_name.startswith("nqopen_"):
        raise ValueError(f"Experiment '{experiment_name}' is not an nqopen_* experiment")

    save_dir = Path(
        f"figures/steering_hallucination/{experiment_name}/{k_type}/top_k={k}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    result_path = save_dir / f"alpha={alpha}.json"
    samples_path = save_dir / f"samples_alpha={alpha}.txt"

    questions, best_answers, correct_lists = load_nqopen_questions()
    n_total = len(questions)
    print(f"Loaded {n_total} NQOpen questions for {experiment_name}")

    if result_path.exists():
        with open(result_path, "r") as f:
            metrics = json.load(f)

        if not samples_path.exists():
            sample_q = questions[:N_SAMPLE_GENERATIONS]
            sample_best = best_answers[:N_SAMPLE_GENERATIONS]
            sample_correct = correct_lists[:N_SAMPLE_GENERATIONS]

            sample_generations, _ = generate_with_steering_batch(
                model=model,
                sae=sae,
                prompts=sample_q,
                feature_indices=feature_indices,
                steering_strength=alpha,
                max_new_tokens=max_new_tokens,
                use_global_max_act=True,
            )

            save_sample_generations_hallucination(
                samples_path,
                experiment_name,
                k_type,
                k,
                alpha,
                sample_q,
                sample_best,
                sample_correct,
                sample_generations,
            )

        return metrics

    n_best_correct = 0
    n_any_correct = 0

    for i in tqdm(range(0, n_total, batch_size),
                  desc=f"Hallucination eval: k={k}, {k_type}, alpha={alpha}"):

        batch_q = questions[i:i + batch_size]
        batch_best = best_answers[i:i + batch_size]
        batch_correct = correct_lists[i:i + batch_size]

        generations, _ = generate_with_steering_batch(
            model=model,
            sae=sae,
            prompts=batch_q,
            feature_indices=feature_indices,
            steering_strength=alpha,
            max_new_tokens=max_new_tokens,
            use_global_max_act=True,
        )

        for gen_text, best_ans, correct_ans_list in zip(
            generations, batch_best, batch_correct
        ):
            g = gen_text.lower()
            best_hit = False
            if isinstance(best_ans, str) and best_ans.strip():
                best_hit = best_ans.lower() in g

            any_hit = best_hit
            if not any_hit:
                for ans in correct_ans_list:
                    if ans and ans.lower() in g:
                        any_hit = True
                        break

            if best_hit:
                n_best_correct += 1
            if any_hit:
                n_any_correct += 1

    best_acc = n_best_correct / n_total
    any_acc = n_any_correct / n_total

    metrics = {
        "best_accuracy": best_acc,
        "any_accuracy": any_acc,
        "n_total": n_total,
    }

    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if not samples_path.exists():
        sample_q = questions[:N_SAMPLE_GENERATIONS]
        sample_best = best_answers[:N_SAMPLE_GENERATIONS]
        sample_correct = correct_lists[:N_SAMPLE_GENERATIONS]

        sample_generations, _ = generate_with_steering_batch(
            model=model,
            sae=sae,
            prompts=sample_q,
            feature_indices=feature_indices,
            steering_strength=alpha,
            max_new_tokens=max_new_tokens,
            use_global_max_act=True,
        )

        save_sample_generations_hallucination(
            samples_path,
            experiment_name,
            k_type,
            k,
            alpha,
            sample_q,
            sample_best,
            sample_correct,
            sample_generations,
        )

    return metrics


def plot_hallucination_curves(experiment_name, k_type, results):

    save_dir = Path(f"figures/steering_hallucination/{experiment_name}/{k_type}")
    save_dir.mkdir(parents=True, exist_ok=True)

    ks = [1, 5, 10, "all"]
    alphas = ["baseline", 0.5, 1.0, 3.0]
    metrics = ["best_accuracy", "any_accuracy"]

    for metric in metrics:
        for alpha in alphas:
            plt.figure(figsize=(8, 5))

            small_ks = [1, 5, 10]
            y_small = [_get_metric(results, k, alpha, metric) for k in small_ks]
            y_all   = _get_metric(results, "all", alpha, metric)


            plt.plot(small_ks, y_small, marker="o", label="k = 1,5,10")
            plt.scatter([15], [y_all], color="red", label="k = all")

            plt.title(f"{metric} vs k | {experiment_name} | {k_type} | alpha={alpha}")
            plt.xlabel("k (axis break before 'all')")
            plt.ylabel(metric)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_dir / f"{metric}_vs_k_alpha={alpha}.png", dpi=200)
            plt.close()

    for metric in metrics:
        for k in ks:
            plt.figure(figsize=(8, 5))
            y_vals = [results[k][alpha][metric] for alpha in alphas]

            plt.plot(alphas, y_vals, marker="o")
            plt.title(f"{metric} vs alpha | {experiment_name} | {k_type} | k={k}")
            plt.xlabel("alpha")
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(save_dir / f"{metric}_vs_alpha_k={k}.png", dpi=200)
            plt.close()

def load_or_generate_baseline_generations(
    experiment_name,
    model,
    sae,
    prompts,
    save_dir,
    max_new_tokens=80,
    batch_size=8,
):


    save_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = save_dir / "baseline.json"
    baseline_samples_path = save_dir / "baseline_samples.txt"

    if baseline_path.exists() and baseline_samples_path.exists():
        with open(baseline_path, "r") as f:
            baseline_data = json.load(f)
        return baseline_data

    print(f"[BASELINE] Generating baseline outputs for {experiment_name}...")

    toxicity_scores = []
    sample_prompts = prompts[:N_SAMPLE_GENERATIONS]
    sample_generations = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Baseline generation"):
        batch = prompts[i:i+batch_size]

        tokens = model.to_tokens(
            batch,
            prepend_bos=sae.cfg.metadata.prepend_bos
        ).to(DEVICE)

        out_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            stop_at_eos=True,
        )

        generations = [model.tokenizer.decode(o) for o in out_tokens]

        for text in generations:
            toxicity_scores.append(get_toxicity_score(text))

        if len(sample_generations) < N_SAMPLE_GENERATIONS:
            take = min(N_SAMPLE_GENERATIONS - len(sample_generations), len(generations))
            sample_generations.extend(generations[:take])

    avg_score = float(sum(toxicity_scores) / len(toxicity_scores))
    metrics = {"avg_toxicity": avg_score}

    with open(baseline_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(baseline_samples_path, "w", encoding="utf-8") as f:
        for i, (p, g) in enumerate(zip(sample_prompts, sample_generations)):
            f.write(f"=== BASELINE SAMPLE {i+1} ===\n")
            f.write("PROMPT:\n" + p + "\n\n")
            f.write("GENERATION:\n" + g + "\n\n")
            f.write("-" * 80 + "\n\n")

    return metrics



def load_or_generate_hallucination_baseline(
    experiment_name,
    model,
    sae,
    questions,
    best_answers,
    correct_lists,
    save_dir,
    max_new_tokens=80,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = save_dir / "baseline.json"
    baseline_samples_path = save_dir / "baseline_samples.txt"

    if baseline_path.exists() and baseline_samples_path.exists():
        with open(baseline_path, "r") as f:
            return json.load(f)

    print(f"[BASELINE] Generating hallucination baseline for {experiment_name}...")

    outputs = []
    for i in tqdm(range(0, len(questions), 8), desc="Baseline generation"):
        batch = questions[i:i+8]
        tokens = model.to_tokens(
            batch,
            prepend_bos=sae.cfg.metadata.prepend_bos
        ).to(DEVICE)

        out_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            stop_at_eos=True
            # prepend_bos=sae.cfg.metadata.prepend_bos,
        )

        outputs.extend([model.tokenizer.decode(o) for o in out_tokens])

    n_total = len(questions)
    n_best_correct = 0
    n_any_correct = 0

    for gen, best_ans, correct in zip(outputs, best_answers, correct_lists):
        L = gen.lower()

        best_hit = isinstance(best_ans, str) and best_ans.lower() in L
        any_hit = best_hit or any((ans.lower() in L) for ans in correct)

        if best_hit:
            n_best_correct += 1
        if any_hit:
            n_any_correct += 1

    metrics = {
        "best_accuracy": n_best_correct / n_total,
        "any_accuracy": n_any_correct / n_total,
        "n_total": n_total,
    }

    with open(baseline_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(baseline_samples_path, "w", encoding="utf-8") as f:
        for i, (q, g) in enumerate(
            zip(
                questions[:N_SAMPLE_GENERATIONS],
                outputs[:N_SAMPLE_GENERATIONS],
            )
        ):
            f.write(f"=== BASELINE SAMPLE {i+1} ===\n")
            f.write("QUESTION:\n" + q + "\n\n")
            f.write("GENERATION:\n" + g + "\n\n")
            f.write("-" * 80 + "\n\n")

    return metrics


def build_feature_indices(k_type, sorted_pos, sorted_neg, k):

    if k == "all":
        if k_type == "positive":
            return [(f.item(), -1) for f in sorted_pos]
        elif k_type == "negative":
            return [(f.item(), +1) for f in sorted_neg]
        else:
            return ([(f.item(), -1) for f in sorted_pos] +
                    [(f.item(), +1) for f in sorted_neg])
    else:
        if k_type == "positive":
            return [(f.item(), -1) for f in sorted_pos[:k]]
        elif k_type == "negative":
            return [(f.item(), +1) for f in sorted_neg[:k]]
        else:
            return ([(f.item(), -1) for f in sorted_pos[:k]] +
                    [(f.item(), +1) for f in sorted_neg[:k]])

def run_steering_analysis(
    experiment_name,
    model,
    sae,
    pos_mean,
    neg_mean,
    max_new_tokens=80,
):
    print(f"\n=== Steering Analysis for {experiment_name} ===")

    diff = pos_mean - neg_mean
    sorted_pos = diff.argsort(descending=True)
    sorted_neg = diff.argsort(descending=False)

    ks = [5]
    alphas = [1.0, 5.0]
    k_types = ["positive", "negative", "both"]

    is_toxicity = experiment_name.startswith("real_toxicity_")
    is_hallucination = experiment_name.startswith("nqopen_")

    all_results = {}

    print("Computing baseline once ...")

    if is_toxicity:
        threshold = float(experiment_name.split("_")[-1])
        prompts = load_real_toxicity_prompts(threshold)

        baseline_save_dir = Path(f"figures/steering_toxicity/{experiment_name}/baseline")
        
        baseline = load_or_generate_baseline_generations(
            experiment_name,
            model,
            sae,
            prompts,
            baseline_save_dir,
            max_new_tokens=max_new_tokens,
        )

    elif is_hallucination:
        questions, best_answers, correct_lists = load_nqopen_questions()

        baseline_save_dir = Path(f"figures/steering_hallucination/{experiment_name}/baseline")
        baseline = load_or_generate_hallucination_baseline(
            experiment_name,
            model,
            sae,
            questions,
            best_answers,
            correct_lists,
            baseline_save_dir,
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(f"Unknown experiment type for {experiment_name}")


    for k_type in k_types:
        print(f"\n====== k_type={k_type} ======")
        results = {}

        for k in ks:

            feature_indices = build_feature_indices(k_type, sorted_pos, sorted_neg, k)

            results[k] = {}
            if is_toxicity:
                results[k]["baseline"] = baseline["avg_toxicity"]
            elif is_hallucination:
                results[k]["baseline"] = baseline

            for alpha in alphas:

                if alpha == "baseline":
                    # no steering
                    feature_indices_used = []
                    alpha_used = 0.0
                else:
                    feature_indices_used = feature_indices
                    alpha_used = alpha

                if is_toxicity:
                    avg_tox = run_toxic_analysis(
                        experiment_name=experiment_name,
                        k_type=k_type,
                        k=k,
                        alpha=alpha_used,
                        feature_indices=feature_indices_used,
                        model=model,
                        sae=sae,
                        max_new_tokens=max_new_tokens,
                    )
                    results[k][alpha] = avg_tox

                elif is_hallucination:
                    metrics = run_hallucination_analysis(
                        experiment_name=experiment_name,
                        k_type=k_type,
                        k=k,
                        alpha=alpha_used,
                        feature_indices=feature_indices_used,
                        model=model,
                        sae=sae,
                        max_new_tokens=max_new_tokens,
                    )
                    results[k][alpha] = metrics


                else:
                    results[k][alpha] = None

        if is_toxicity:
            plot_toxicity_curves(experiment_name, k_type, results)

        if is_hallucination:
            plot_hallucination_curves(experiment_name, k_type, results)

        all_results[k_type] = results

    return all_results
