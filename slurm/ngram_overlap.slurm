#!/bin/bash
#SBATCH --job-name=ngram-overlap
#SBATCH --account=bcyi-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/bcyi/eergun/silo-knn/logs/%x-%j.out
#
# Per-document 13-gram overlap of a domain's val/test with its training data (scripts/ngram_overlap.py).
# CPU-only work; it runs on the GPU partition because the CPU partition is usually full.
#
#   sbatch slurm/ngram_overlap.slurm                                     # Enron_Emails
set -eo pipefail   # no -u: module and venv activation scripts reference unset variables

REPO=/work/nvme/bcyi/eergun/silo-lm
SUBSET=${SUBSET:-Enron_Emails}
WORK=${WORK:-/work/hdd/bcyi/eergun/silo-knn}   # data/ is resolved relative to this directory
OUT_DIR=${OUT_DIR:-data/the-pile/$SUBSET-overlap}
VENV=/work/hdd/bcyi/eergun/silo-knn/venv

module load pytorch-conda
source "$VENV/bin/activate"
cd "$WORK"
t0=$(date +%s)
python3 "$REPO/scripts/ngram_overlap.py" --subset "$SUBSET" --out_dir "$OUT_DIR"
echo "overlap done: $(( $(date +%s) - t0 ))s"
