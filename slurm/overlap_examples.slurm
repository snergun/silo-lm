#!/bin/bash
#SBATCH --job-name=overlap-examples
#SBATCH --account=bcyi-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=1
#SBATCH --mem=24g
#SBATCH --time=00:20:00
#SBATCH --output=/work/hdd/bcyi/eergun/silo-knn/logs/%x-%j.out
#
# Classify and show examples of the val/test documents with >= 50% 13-gram overlap with train
# (scripts/overlap_examples.py). CPU-only work; it runs on the GPU partition because the CPU
# partition is usually full.
#
#   sbatch --mem=8g --time=00:10:00 slurm/overlap_examples.slurm         # Enron_Emails
#   SUBSET=cc-news WORK=/work/nvme/bcyi/eergun/silo-knn-ccnews OVERLAP_DIR=data/cc-news \
#       sbatch slurm/overlap_examples.slurm
set -eo pipefail   # no -u: module and venv activation scripts reference unset variables

REPO=/work/nvme/bcyi/eergun/silo-lm
SUBSET=${SUBSET:-Enron_Emails}
WORK=${WORK:-/work/hdd/bcyi/eergun/silo-knn}   # data/ is resolved relative to this directory
OVERLAP_DIR=${OVERLAP_DIR:-data/the-pile/$SUBSET-overlap}
OUT_DIR=${OUT_DIR:-data/$SUBSET-overlap-examples}
VENV=/work/hdd/bcyi/eergun/silo-knn/venv

module load pytorch-conda
source "$VENV/bin/activate"
cd "$WORK"
t0=$(date +%s)
python3 "$REPO/scripts/overlap_examples.py" --subset "$SUBSET" --overlap_dir "$OVERLAP_DIR" --out_dir "$OUT_DIR"
echo "done: $(( $(date +%s) - t0 ))s"
