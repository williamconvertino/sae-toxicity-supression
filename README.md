# SAE-based Toxicity Suppression for LLMs

This project implements Sparse Autoencoder (SAE) based steering to reduce toxic outputs from Large Language Models. Developed for ECE685D Project 2.

## Overview

1. Discover interpretable features in LLM activations using pre-trained SAEs
2. Identify harmful (F+) vs. protective (F-) features by comparing toxic and benign text
3. Steer the model at inference time to suppress toxic outputs

## Installation

```bash
# Create environment
conda create -n sae-toxicity python=3.10 -y
conda activate sae-toxicity

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (required for Gemma model)
huggingface-cli login
```

## Usage

### Step 1: Feature Discovery

```bash
python identify_features.py
```

Outputs: `real_toxicity/`, `nqopen_onlybest/`, `nqopen_allneg/`

### Step 2: Steering Experiments

```bash
# Safety steering
python steering.py --task safety --max-samples 100

# Hallucination steering
python steering.py --task hallucination --max-samples 100

# Both
python steering.py --task both --max-samples 100
```

Options:
- `--task`: `safety`, `hallucination`, or `both`
- `--k`: Number of top features (default: 20)
- `--max-samples`: Samples to evaluate (default: 100)

### Step 3: Evaluate Results

```bash
python evaluate_toxicity.py --results steering_results/safety/generation_results.json
```

## Project Structure

```
├── util.py                 # Model loading and data utilities
├── identify_features.py    # Feature discovery
├── steering.py             # Steering experiments
├── evaluate_toxicity.py    # Toxicity evaluation
├── train_classifiers.py    # Detection classifiers
└── requirements.txt        # Dependencies
```

## Configuration

### GPU Selection

Modify `DEVICE` in each script or `load_model()` in `util.py`:

```python
DEVICE = torch.device("cuda:0")
```

### Model

- Model: `google/gemma-2b-it`
- SAE: `gemma-2b-it-res-jb` at layer 12

## References

- [SAE-Lens](https://github.com/jbloomAus/SAELens)
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens)
- [RealToxicityPrompts](https://huggingface.co/datasets/allenai/real-toxicity-prompts)
- [NQ-Open](https://huggingface.co/datasets/baonn/nqopen)
- [Anthropic HH](https://huggingface.co/datasets/Anthropic/hh-rlhf)
