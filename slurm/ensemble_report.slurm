#!/bin/bash
#SBATCH --job-name=ens-report
#SBATCH --account=bcyi-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g
#SBATCH --time=00:20:00
#SBATCH --output=/work/hdd/bcyi/eergun/silo-knn/logs/%x-%j.out
#
# Datastore-overlap breakdown of every finished combiner run (ensemble/overlap_breakdown.py, which
# loads per-token test NLLs), then results/pythia_knn_ensemble.md (ensemble/report.py). Domains whose
# runs have not started are skipped. CPU-only; on the GPU partition because the CPU queue is slow.
#
#   sbatch slurm/ensemble_report.slurm
set -eo pipefail

REPO=/work/nvme/bcyi/eergun/silo-lm
VENV=/work/hdd/bcyi/eergun/silo-knn/venv
module load pytorch-conda
source "$VENV/bin/activate"
export PYTHONPATH=$REPO:$REPO/scripts

for spec in "Enron_Emails /work/hdd/bcyi/eergun/silo-knn 8" "cc-news /work/nvme/bcyi/eergun/silo-knn-ccnews 9"; do
    set -- $spec
    if [ -f "$2/out/pythia-1.4B/ensemble/$1/fixed/.done" ]; then
        t0=$(date +%s)
        python3 -m ensemble.overlap_breakdown --work "$2" --subset "$1" --last_idx $3
        echo "breakdown $1: $(( $(date +%s) - t0 ))s"
    fi
done
cd "$REPO"
t0=$(date +%s)
python3 ensemble/report.py --out results/pythia_knn_ensemble.md
echo "report: $(( $(date +%s) - t0 ))s"
