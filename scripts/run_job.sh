#!/bin/bash
#SBATCH --job-name=run_toxicity_analysis
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --exclude=dcc-h200-gpu-04,dcc-h200-gpu-05
#SBATCH --nodelist=dcc-h200-gpu-01,dcc-h200-gpu-02,dcc-h200-gpu-03,dcc-h200-gpu-06,dcc-h200-gpu-07
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --account=h200ea
#SBATCH --partition=h200ea
#SBATCH --output=../logs/%x-%j.out
#SBATCH --error=../logs/%x-%j.err

if [ -f ./env.sh ]; then
    echo "🔹 Sourcing ./env.sh"
    source ./env.sh
fi

export WANDB_MODE=offline

cd ../

python main.py "$@"