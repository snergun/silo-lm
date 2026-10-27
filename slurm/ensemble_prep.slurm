#!/bin/bash
#SBATCH --job-name=ens-prep
#SBATCH --account=bcyi-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bcyi/eergun/silo-knn/logs/%x-%j.out
#
# Prepares a domain for the LM + kNN combiners (ensemble/prepare.py): fixed kNN-LM chosen on the fit
# slice of val, compact neighbor cache, Pythia's embed_out, and parity checks against scripts/main.py.
# Runs the synthetic combiner tests first; SMOKE=1 then trains the non-linear arm for 200 steps.
#
#   sbatch --job-name=enron-ens-prep --export=ALL,SMOKE=1 slurm/ensemble_prep.slurm
#   sbatch --job-name=ccnews-ens-prep --export=ALL,SUBSET=cc-news,WORK=/work/nvme/bcyi/eergun/silo-knn-ccnews,LAST_IDX=9 \
#          slurm/ensemble_prep.slurm
set -eo pipefail   # no -u: module and venv activation scripts reference unset variables

REPO=/work/nvme/bcyi/eergun/silo-lm
SUBSET=${SUBSET:-Enron_Emails}
WORK=${WORK:-/work/hdd/bcyi/eergun/silo-knn}
LAST_IDX=${LAST_IDX:-8}
VENV=/work/hdd/bcyi/eergun/silo-knn/venv

module load pytorch-conda
source "$VENV/bin/activate"
export PYTHONPATH=$REPO:$REPO/scripts
cd "$WORK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

t0=$(date +%s)
python3 -m ensemble.test_combiners
echo "tests: $(( $(date +%s) - t0 ))s"

t0=$(date +%s)
python3 -m ensemble.prepare --work "$WORK" --subset "$SUBSET" --last_idx $LAST_IDX
echo "prepare: $(( $(date +%s) - t0 ))s"

if [ "${SMOKE:-0}" = "1" ]; then
    t0=$(date +%s)
    python3 -m ensemble.train --work "$WORK" --subset "$SUBSET" --last_idx $LAST_IDX --arm nonlinear \
        --name smoke_nonlinear --max_steps 200 --eval_interval 50
    echo "smoke: $(( $(date +%s) - t0 ))s"
fi
echo "prep done"
