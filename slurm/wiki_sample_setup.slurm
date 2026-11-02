#!/bin/bash
#SBATCH --job-name=wiki-sample
#SBATCH --account=bcyi-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=2
#SBATCH --mem=32g
#SBATCH --time=00:20:00
#SBATCH --output=/work/hdd/bcyi/eergun/silo-knn/logs/%x-%j.out
#
# Evaluation root for a partial Wikipedia datastore: a ~1.1M-token random document sample of val and
# test (scripts/sample_eval_docs.py) in $OUT_ROOT. The root must already hold, under out/, a hard link
# to the pinned index (pythia-1.4B/train/...-[0K-<end>K].index, so the build chain's
# --remove_prev_index cannot delete it) and symlinks to the train tokens and shard targets
# (neoX/train/). Then: slurm/eval_knn.slurm and the ensemble jobs with WORK=$OUT_ROOT.
# CPU-only; on the GPU partition because the CPU queue is slow.
#
#   sbatch slurm/wiki_sample_setup.slurm
set -eo pipefail

REPO=/work/nvme/bcyi/eergun/silo-lm
SUBSET=${SUBSET:-Wikipedia_(en)}
SRC_ROOT=${SRC_ROOT:-/work/hdd/bcyi/eergun/silo-knn}
OUT_ROOT=${OUT_ROOT:-/work/hdd/bcyi/eergun/silo-knn-wiki}
VENV=/work/hdd/bcyi/eergun/silo-knn/venv

module load pytorch-conda
source "$VENV/bin/activate"
t0=$(date +%s)
python3 "$REPO/scripts/sample_eval_docs.py" --subset "$SUBSET" --src_root "$SRC_ROOT" --out_root "$OUT_ROOT"
echo "sample: $(( $(date +%s) - t0 ))s"
