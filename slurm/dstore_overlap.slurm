#!/bin/bash
#SBATCH --job-name=dstore-overlap
#SBATCH --account=bcyi-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bcyi/eergun/silo-knn/logs/%x-%j.out
#
# Token 13-gram overlap of val/test with the datastore itself (scripts/dstore_ngram_overlap.py),
# then the kNN-LM report broken down by it (needs slurm/eval_knn.slurm to have finished).
# CPU-only work; it runs on the GPU partition because the CPU partition is usually full.
#
#   sbatch slurm/dstore_overlap.slurm                                    # Enron_Emails, 9 shards
#   sbatch --export=ALL,SUBSET=cc-news,WORK=/work/nvme/bcyi/eergun/silo-knn-ccnews,N_SHARDS=10 slurm/dstore_overlap.slurm
set -eo pipefail   # no -u: module and venv activation scripts reference unset variables

REPO=/work/nvme/bcyi/eergun/silo-lm
SUBSET=${SUBSET:-Enron_Emails}
WORK=${WORK:-/work/hdd/bcyi/eergun/silo-knn}   # data/ and out/ are resolved relative to this directory
N_SHARDS=${N_SHARDS:-9}
OUT_DIR=data/$SUBSET-dstore-overlap
VENV=/work/hdd/bcyi/eergun/silo-knn/venv
N=10   # --max_n_sequences used to build the datastore

module load pytorch-conda
source "$VENV/bin/activate"
cd "$WORK"
t0=$(date +%s)
PYTHONPATH=$REPO python3 "$REPO/scripts/dstore_ngram_overlap.py" --subset "$SUBSET" \
    --max_n_sequences $N --n_shards $N_SHARDS --out_dir "$OUT_DIR"
echo "overlap done: $(( $(date +%s) - t0 ))s"

PYTHONPATH=$REPO python3 "$REPO/scripts/report_knn.py" --subset "$SUBSET" --embed_idx $((N_SHARDS - 1)) \
    --max_n_sequences $N --k 4096 --approximate --overlap_dir "$OUT_DIR" --report_name knn_eval_report_dstore_overlap
echo "report done"
