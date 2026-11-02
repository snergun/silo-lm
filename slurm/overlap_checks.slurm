#!/bin/bash
#SBATCH --job-name=overlap-checks
#SBATCH --account=bcyi-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=1
#SBATCH --mem=48g
#SBATCH --time=00:30:00
#SBATCH --output=/work/hdd/bcyi/eergun/silo-knn/logs/%x-%j.out
#
# Word n-gram overlap (n = 8, 13, 20, 50) of val/test with the training split and with the decoded
# datastore, plus the other domain's val/test as a chance baseline (scripts/overlap_checks.py).
# Needs slurm/dstore_overlap.slurm to have run. CPU-only work; it runs on the GPU partition because
# the CPU partition is usually full.
#
#   sbatch --mem=16g --time=00:15:00 slurm/overlap_checks.slurm         # Enron_Emails
#   SUBSET=cc-news WORK=/work/nvme/bcyi/eergun/silo-knn-ccnews N_SHARDS=10 STORED=data/cc-news \
#       OTHER_SUBSET=Enron_Emails OTHER_WORK=/work/hdd/bcyi/eergun/silo-knn sbatch slurm/overlap_checks.slurm
set -eo pipefail   # no -u: module and venv activation scripts reference unset variables

REPO=/work/nvme/bcyi/eergun/silo-lm
SUBSET=${SUBSET:-Enron_Emails}
WORK=${WORK:-/work/hdd/bcyi/eergun/silo-knn}   # data/ and out/ are resolved relative to this directory
N_SHARDS=${N_SHARDS:-9}
STORED=${STORED:-data/the-pile/$SUBSET-overlap}
OTHER_SUBSET=${OTHER_SUBSET:-cc-news}
OTHER_WORK=${OTHER_WORK:-/work/nvme/bcyi/eergun/silo-knn-ccnews}
OUT_DIR=data/$SUBSET-overlap-checks
VENV=/work/hdd/bcyi/eergun/silo-knn/venv

module load pytorch-conda
source "$VENV/bin/activate"
cd "$WORK"
t0=$(date +%s)
python3 "$REPO/scripts/overlap_checks.py" --subset "$SUBSET" --n_shards $N_SHARDS --stored_overlap_dir "$STORED" \
    --other_subset "$OTHER_SUBSET" --other_work "$OTHER_WORK" --out_dir "$OUT_DIR"
echo "done: $(( $(date +%s) - t0 ))s"
