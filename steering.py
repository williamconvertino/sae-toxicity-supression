"""
SAE-based Steering for Toxicity and Hallucination Reduction.

This module implements steering vectors from SAE features to:
1. Reduce toxic/unsafe continuations (safety steering)
2. Reduce hallucinated/incorrect answers (hallucination steering)

Approach:
- Compute steering direction from F+ (undesirable) and F- (protective) features
- Apply steering by modifying residual stream activations during generation
- Evaluate before/after rates on benchmark datasets
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from functools import partial

from util import load_model, load_real_toxicity, load_nqopen, TextDataset

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def load_feature_sets(
    analysis_name: str,
    k: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load F+ and F- feature indices from saved mean activations.
    
    Args:
        analysis_name: Directory containing positive.pt and negative.pt
        k: Number of top features to select
        
    Returns:
        F_plus: indices of features correlated with undesirable outcome
        F_minus: indices of features that suppress undesirable outcome
        diff: full difference vector (pos_mean - neg_mean)
    """
    save_dir = Path(analysis_name)
    pos_path = save_dir / "positive.pt"
    neg_path = save_dir / "negative.pt"
    
    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError(
            f"Mean activation files not found for '{analysis_name}'. "
            f"Run identify_features.py first."
        )
    
    pos_mean = torch.load(pos_path, map_location="cpu")
    neg_mean = torch.load(neg_path, map_location="cpu")
    
    diff = pos_mean - neg_mean
    
    F_plus = diff.topk(k).indices      # features activated by undesirable
    F_minus = (-diff).topk(k).indices  # features that suppress undesirable
    
    return F_plus, F_minus, diff


def compute_steering_vector(
    sae,
    F_plus: torch.Tensor,
    F_minus: torch.Tensor,
    alpha_plus: float = 1.0,
    alpha_minus: float = 1.0,
) -> torch.Tensor:
    """
    Compute steering vector in residual stream space.
    
    The steering direction pushes F+ features down and F- features up.
    We use the SAE decoder weights to map from SAE feature space to residual space.
    
    steering_vec = -alpha_plus * sum(W_dec[f+]) + alpha_minus * sum(W_dec[f-])
    
    Args:
        sae: Sparse Autoencoder with W_dec attribute
        F_plus: indices of features to suppress
        F_minus: indices of features to boost
        alpha_plus: scaling for F+ suppression
        alpha_minus: scaling for F- boosting
        
    Returns:
        steering_vec: (d_model,) tensor representing steering direction
    """
    W_dec = sae.W_dec  # (d_sae, d_model)
    
    # Get decoder vectors for F+ and F- features
    vec_plus = W_dec[F_plus].sum(dim=0)   # sum of decoder vectors for F+
    vec_minus = W_dec[F_minus].sum(dim=0) # sum of decoder vectors for F-
    
    # Steering direction: suppress F+, boost F-
    steering_vec = -alpha_plus * vec_plus + alpha_minus * vec_minus
    
    # Normalize for stability
    steering_vec = F.normalize(steering_vec, dim=0)
    
    return steering_vec


class SteeringHook:
    """
    Hook to apply steering vector to residual stream during generation.
    """
    
    def __init__(
        self,
        steering_vec: torch.Tensor,
        strength: float = 1.0,
        apply_to_last_token_only: bool = True,
    ):
        """
        Args:
            steering_vec: (d_model,) steering direction
            strength: scalar multiplier for steering
            apply_to_last_token_only: if True, only modify last token's activations
        """
        self.steering_vec = steering_vec.to(DEVICE)
        self.strength = strength
        self.apply_to_last_token_only = apply_to_last_token_only
    
    def __call__(self, activations, hook):
        """
        Hook function compatible with TransformerLens.
        
        Args:
            activations: (batch, seq_len, d_model) residual stream activations
            hook: HookPoint object
            
        Returns:
            Modified activations
        """
        if self.apply_to_last_token_only:
            # Only modify the last token position
            activations[:, -1, :] = activations[:, -1, :] + self.strength * self.steering_vec
        else:
            # Modify all positions
            activations = activations + self.strength * self.steering_vec
        
        return activations


def generate_with_steering(
    model,
    sae,
    prompt: str,
    steering_vec: Optional[torch.Tensor] = None,
    strength: float = 1.0,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """
    Generate text with optional steering applied.
    
    Args:
        model: HookedTransformer
        sae: SAE (used to get hook name)
        prompt: input text
        steering_vec: (d_model,) steering direction, or None for no steering
        strength: steering strength
        max_new_tokens: maximum tokens to generate
        temperature: sampling temperature
        do_sample: whether to sample or use greedy decoding
        
    Returns:
        Generated text (continuation only, not including prompt)
    """
    tok = model.tokenizer
    hook_name = sae.cfg.metadata.hook_name
    
    # Tokenize
    input_ids = tok.encode(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = input_ids.shape[1]
    
    # Setup hook if steering
    hooks = []
    if steering_vec is not None:
        hook = SteeringHook(steering_vec, strength=strength, apply_to_last_token_only=True)
        hooks = [(hook_name, hook)]
    
    # Generate token by token with hooks
    generated_ids = input_ids.clone()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            if hooks:
                # Run with steering hook
                logits = model.run_with_hooks(
                    generated_ids,
                    fwd_hooks=hooks,
                    return_type="logits"
                )
            else:
                logits = model(generated_ids)
            
            # Get next token logits
            next_logits = logits[:, -1, :]
            
            if do_sample and temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop if EOS
            if next_token.item() == tok.eos_token_id:
                break
    
    # Decode only the generated part
    full_text = tok.decode(generated_ids[0], skip_special_tokens=True)
    prompt_text = tok.decode(input_ids[0], skip_special_tokens=True)
    
    # Return continuation only
    continuation = full_text[len(prompt_text):].strip()
    return continuation


def evaluate_toxicity_steering(
    model,
    sae,
    steering_vec: torch.Tensor,
    prompts: List[str],
    strengths: List[float] = [0.0, 0.5, 1.0, 2.0],
    max_new_tokens: int = 50,
    max_samples: int = 100,
) -> Dict[str, List[Dict]]:
    """
    Evaluate steering effect on toxicity using RealToxicityPrompts.
    
    This function generates continuations with different steering strengths
    and collects them for later toxicity scoring.
    
    Args:
        model: HookedTransformer
        sae: SAE
        steering_vec: steering direction
        prompts: list of toxic prompts to continue
        strengths: list of steering strengths to test
        max_new_tokens: tokens to generate per prompt
        max_samples: maximum prompts to evaluate
        
    Returns:
        Dict mapping strength -> list of {prompt, continuation} dicts
    """
    results = {str(s): [] for s in strengths}
    
    prompts = prompts[:max_samples]
    
    for prompt in tqdm(prompts, desc="Generating continuations"):
        for strength in strengths:
            if strength == 0.0:
                vec = None
            else:
                vec = steering_vec
            
            try:
                continuation = generate_with_steering(
                    model=model,
                    sae=sae,
                    prompt=prompt,
                    steering_vec=vec,
                    strength=strength,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                )
            except Exception as e:
                print(f"Generation error: {e}")
                continuation = "[ERROR]"
            
            results[str(strength)].append({
                "prompt": prompt,
                "continuation": continuation,
            })
    
    return results


def evaluate_hallucination_steering(
    model,
    sae,
    steering_vec: torch.Tensor,
    questions: List[str],
    correct_answers: List[List[str]],
    strengths: List[float] = [0.0, 0.5, 1.0, 2.0],
    max_new_tokens: int = 30,
    max_samples: int = 100,
) -> Dict[str, List[Dict]]:
    """
    Evaluate steering effect on hallucination using NQ-Open.
    
    Args:
        model: HookedTransformer
        sae: SAE
        steering_vec: steering direction
        questions: list of questions
        correct_answers: list of lists of acceptable answers per question
        strengths: steering strengths to test
        max_new_tokens: tokens to generate
        max_samples: max questions to evaluate
        
    Returns:
        Dict mapping strength -> list of {question, generated, correct_answers} dicts
    """
    results = {str(s): [] for s in strengths}
    
    questions = questions[:max_samples]
    correct_answers = correct_answers[:max_samples]
    
    for question, answers in tqdm(zip(questions, correct_answers), 
                                   total=len(questions),
                                   desc="Generating answers"):
        prompt = f"Question: {question}\nAnswer:"
        
        for strength in strengths:
            if strength == 0.0:
                vec = None
            else:
                vec = steering_vec
            
            try:
                generated = generate_with_steering(
                    model=model,
                    sae=sae,
                    prompt=prompt,
                    steering_vec=vec,
                    strength=strength,
                    max_new_tokens=max_new_tokens,
                    temperature=0.3,  # Lower temp for factual answers
                    do_sample=True,
                )
            except Exception as e:
                print(f"Generation error: {e}")
                generated = "[ERROR]"
            
            results[str(strength)].append({
                "question": question,
                "generated": generated,
                "correct_answers": answers,
            })
    
    return results


def simple_answer_match(generated: str, correct_answers: List[str]) -> bool:
    """
    Simple exact match check for hallucination evaluation.
    Returns True if generated answer contains any correct answer.
    """
    generated_lower = generated.lower().strip()
    for ans in correct_answers:
        if ans.lower().strip() in generated_lower:
            return True
    return False


def compute_accuracy_by_strength(
    results: Dict[str, List[Dict]]
) -> Dict[str, float]:
    """
    Compute accuracy for each steering strength.
    """
    accuracies = {}
    for strength, samples in results.items():
        correct = 0
        total = 0
        for sample in samples:
            if "correct_answers" in sample:
                if simple_answer_match(sample["generated"], sample["correct_answers"]):
                    correct += 1
                total += 1
        accuracies[strength] = correct / total if total > 0 else 0.0
    return accuracies


def load_nqopen_for_steering():
    """
    Load NQ-Open dataset in format suitable for steering evaluation.
    Returns (questions, correct_answers_lists)
    """
    from datasets import load_dataset
    
    print("Loading NQ-Open for steering evaluation...")
    ds = load_dataset("baonn/nqopen", "generation", split="validation")
    
    questions = []
    correct_answers = []
    
    for row in ds:
        q = row["question"]
        
        # Get all correct answers
        answers = []
        best = row.get("best_answer", "")
        if best and best.strip():
            answers.append(best.strip())
        
        all_correct = row.get("correct_answers", "")
        if all_correct:
            for ans in all_correct.split(";"):
                ans = ans.strip()
                if ans and ans not in answers:
                    answers.append(ans)
        
        if answers:  # Only include if we have at least one correct answer
            questions.append(q)
            correct_answers.append(answers)
    
    return questions, correct_answers


def load_rtp_prompts_for_steering(toxicity_threshold: float = 0.5, max_size: int = 500):
    """
    Load RealToxicityPrompts for steering evaluation.
    Returns list of toxic prompts.
    """
    from datasets import load_dataset
    
    print("Loading RealToxicityPrompts for steering evaluation...")
    rtp = load_dataset("allenai/real-toxicity-prompts", split="train")
    
    prompts = []
    
    for row in rtp:
        p = row["prompt"]
        
        if isinstance(p, dict):
            toxicity = p.get("toxicity")
            text = p.get("text", "")
            
            if toxicity is not None and toxicity >= toxicity_threshold:
                if text and text.strip():
                    prompts.append(text.strip())
        
        if len(prompts) >= max_size:
            break
    
    print(f"Loaded {len(prompts)} toxic prompts")
    return prompts


def run_safety_steering_experiment(
    model,
    sae,
    analysis_name: str = "real_toxicity",
    k: int = 20,
    strengths: List[float] = [0.0, 0.5, 1.0, 2.0, 3.0],
    max_samples: int = 100,
    output_dir: str = "steering_results",
):
    """
    Run full safety steering experiment on RealToxicityPrompts.
    """
    out_path = Path(output_dir) / "safety"
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load F+ and F-
    print(f"\nLoading features from '{analysis_name}'...")
    F_plus, F_minus, diff = load_feature_sets(analysis_name, k=k)
    print(f"F+ indices: {F_plus.tolist()}")
    print(f"F- indices: {F_minus.tolist()}")
    
    # Compute steering vector
    print("\nComputing steering vector...")
    steering_vec = compute_steering_vector(
        sae=sae,
        F_plus=F_plus,
        F_minus=F_minus,
        alpha_plus=1.0,
        alpha_minus=1.0,
    )
    print(f"Steering vector shape: {steering_vec.shape}")
    print(f"Steering vector norm: {steering_vec.norm().item():.4f}")
    
    # Save steering vector
    torch.save(steering_vec, out_path / "steering_vector.pt")
    
    # Load prompts
    prompts = load_rtp_prompts_for_steering(toxicity_threshold=0.5, max_size=max_samples)
    
    # Generate with different strengths
    print(f"\nGenerating continuations with {len(strengths)} steering strengths...")
    results = evaluate_toxicity_steering(
        model=model,
        sae=sae,
        steering_vec=steering_vec,
        prompts=prompts,
        strengths=strengths,
        max_new_tokens=50,
        max_samples=max_samples,
    )
    
    # Save results
    with open(out_path / "generation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {out_path}")
    print("\nSample generations:")
    for strength in ["0.0", "2.0"]:
        if strength in results and results[strength]:
            sample = results[strength][0]
            print(f"\n[Strength={strength}]")
            print(f"  Prompt: {sample['prompt'][:80]}...")
            print(f"  Continuation: {sample['continuation'][:100]}...")
    
    return results


def run_hallucination_steering_experiment(
    model,
    sae,
    analysis_name: str = "nqopen_onlybest",
    k: int = 20,
    strengths: List[float] = [0.0, 0.5, 1.0, 2.0, 3.0],
    max_samples: int = 100,
    output_dir: str = "steering_results",
):
    """
    Run full hallucination steering experiment on NQ-Open.
    """
    out_path = Path(output_dir) / "hallucination"
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load F+ and F-
    print(f"\nLoading features from '{analysis_name}'...")
    F_plus, F_minus, diff = load_feature_sets(analysis_name, k=k)
    print(f"F+ indices: {F_plus.tolist()}")
    print(f"F- indices: {F_minus.tolist()}")
    
    # Compute steering vector
    print("\nComputing steering vector...")
    steering_vec = compute_steering_vector(
        sae=sae,
        F_plus=F_plus,
        F_minus=F_minus,
        alpha_plus=1.0,
        alpha_minus=1.0,
    )
    print(f"Steering vector shape: {steering_vec.shape}")
    
    # Save steering vector
    torch.save(steering_vec, out_path / "steering_vector.pt")
    
    # Load questions
    questions, correct_answers = load_nqopen_for_steering()
    print(f"Loaded {len(questions)} questions with answers")
    
    # Generate with different strengths
    print(f"\nGenerating answers with {len(strengths)} steering strengths...")
    results = evaluate_hallucination_steering(
        model=model,
        sae=sae,
        steering_vec=steering_vec,
        questions=questions,
        correct_answers=correct_answers,
        strengths=strengths,
        max_new_tokens=30,
        max_samples=max_samples,
    )
    
    # Compute accuracy
    accuracies = compute_accuracy_by_strength(results)
    print("\nAccuracy by steering strength:")
    for strength, acc in accuracies.items():
        print(f"  Strength {strength}: {acc:.2%}")
    
    # Save results
    with open(out_path / "generation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(out_path / "accuracies.json", "w") as f:
        json.dump(accuracies, f, indent=2)
    
    print(f"\nResults saved to {out_path}")
    
    return results, accuracies


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SAE-based Steering Experiments")
    parser.add_argument(
        "--task",
        choices=["safety", "hallucination", "both"],
        default="both",
        help="Which steering experiment to run"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of top F+/F- features to use"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="steering_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Load model and SAE
    print("Loading model and SAE...")
    model, sae = load_model()
    model.to(DEVICE)
    sae.to(DEVICE)
    
    if args.task in ["safety", "both"]:
        print("\n" + "="*60)
        print("SAFETY STEERING EXPERIMENT")
        print("="*60)
        run_safety_steering_experiment(
            model=model,
            sae=sae,
            analysis_name="real_toxicity",
            k=args.k,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
        )
    
    if args.task in ["hallucination", "both"]:
        print("\n" + "="*60)
        print("HALLUCINATION STEERING EXPERIMENT")
        print("="*60)
        run_hallucination_steering_experiment(
            model=model,
            sae=sae,
            analysis_name="nqopen_onlybest",
            k=args.k,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
        )
    
    print("\n" + "="*60)
    print("All experiments complete!")
    print("="*60)


if __name__ == "__main__":
    main()

