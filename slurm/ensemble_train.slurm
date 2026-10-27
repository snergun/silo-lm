#!/bin/bash
#SBATCH --job-name=ens-train
#SBATCH --account=bcyi-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=04:00:00
#SBATCH --output=/work/hdd/bcyi/eergun/silo-knn/logs/%x-%A_%a.out
#
# One LM + kNN combiner per array task (ensemble/train.py), after slurm/ensemble_prep.slurm.
# Array index i trains the i-th arm of ARMS. With USE_PCT=1 every arm loads the frozen piecewise
# temperatures from the pct arm's run, so submit that after the pct task has finished:
#
#   jid=$(sbatch --parsable --job-name=enron-ens --array=0-5 slurm/ensemble_train.slurm)
#   sbatch --job-name=enron-ens-pct --array=0-3 --dependency=afterok:${jid}_1 \
#          --export=ALL,USE_PCT=1,ARMS="dawin dawin_cov adaptive nonlinear" slurm/ensemble_train.slurm
#
# For cc-news add SUBSET=cc-news,WORK=/work/nvme/bcyi/eergun/silo-knn-ccnews,LAST_IDX=9 to --export.
# EXTRA_ARGS is appended to ensemble/train.py's arguments, e.g. EXTRA_ARGS="--batch_size 256".
set -eo pipefail

REPO=/work/nvme/bcyi/eergun/silo-lm
SUBSET=${SUBSET:-Enron_Emails}
WORK=${WORK:-/work/hdd/bcyi/eergun/silo-knn}
LAST_IDX=${LAST_IDX:-8}
ARMS=(${ARMS:-fixed pct dawin dawin_cov adaptive nonlinear})
VENV=/work/hdd/bcyi/eergun/silo-knn/venv
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
OUT=$WORK/out/pythia-1.4B/ensemble/$SUBSET

module load pytorch-conda
source "$VENV/bin/activate"
export PYTHONPATH=$REPO:$REPO/scripts
cd "$WORK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PCT_ARGS=""
if [ "${USE_PCT:-0}" = "1" ]; then
    [ -f "$OUT/pct/.done" ] || { echo "no finished pct run in $OUT/pct"; exit 1; }
    PCT_ARGS="--pct $OUT/pct/pct.pt"
fi
echo "$SUBSET: arm $ARM $PCT_ARGS"
t0=$(date +%s)
python3 -m ensemble.train --work "$WORK" --subset "$SUBSET" --last_idx $LAST_IDX --arm $ARM $PCT_ARGS ${EXTRA_ARGS:-}
echo "train $ARM: $(( $(date +%s) - t0 ))s"
