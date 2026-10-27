#!/bin/bash
#SBATCH --job-name=eval-knn
#SBATCH --account=bcyi-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=/work/hdd/bcyi/eergun/silo-knn/logs/%x-%j.out
#
# kNN-LM evaluation of a datastore built by slurm/build_dstore.slurm, following the paper:
# encode val/test -> search the IVFPQ index (nprobe=8, k=4096, approximate distances) ->
# score the (k, temperature, lambda) grid -> choose on val, report test (scripts/report_knn.py).
# Steps whose outputs exist are skipped, so the job can be resubmitted.
#
#   sbatch --job-name=enron-eval slurm/eval_knn.slurm                     # Enron_Emails, 45.0M tokens
#   sbatch --job-name=ccnews-eval --export=ALL,SUBSET=cc-news,WORK=/work/nvme/bcyi/eergun/silo-knn-ccnews,LAST_IDX=9 \
#          slurm/eval_knn.slurm                                          # cc-news, 51.2M tokens
set -eo pipefail   # no -u: module and venv activation scripts reference unset variables

REPO=/work/nvme/bcyi/eergun/silo-lm
SUBSET=${SUBSET:-Enron_Emails}
WORK=${WORK:-/work/hdd/bcyi/eergun/silo-knn}   # data/ and out/ are resolved relative to this directory
LAST_IDX=${LAST_IDX:-8}                         # index of the last shard in the datastore
VENV=/work/hdd/bcyi/eergun/silo-knn/venv
N=10                                            # --max_n_sequences used to build the datastore

module load pytorch-conda
source "$VENV/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK     # FAISS otherwise starts one thread per core of the node
cd "$WORK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "$SUBSET: evaluating datastore [0K-$((N * (LAST_IDX + 1)))K] in $WORK"

for split in val test; do
    t0=$(date +%s)
    PYTHONPATH=$REPO python3 "$REPO/scripts/main.py" --task encode --split $split \
        --subset "$SUBSET" --lm pythia-1.4B --max_seq_length 1024 --stride 512
    echo "encode $split: $(( $(date +%s) - t0 ))s"
done

# val: full (k, temperature, lambda) grid; test: neighbors are still retrieved, but only the
# configuration that was best on val is scored, as in the paper
VAL_GRID="out/pythia-1.4B/train/$SUBSET-val-1024-512-[0K-$((N * (LAST_IDX + 1)))K]-k=4096-p=8_approx.grid.json"
for split in val test; do
    t0=$(date +%s)
    CONFIG_ARGS=""
    [ "$split" = "test" ] && CONFIG_ARGS="--config_from $VAL_GRID"
    PYTHONPATH=$REPO python3 "$REPO/scripts/main.py" --task inference --split train --val_split $split \
        --subset "$SUBSET" --lm pythia-1.4B --max_seq_length 1024 --stride 512 \
        --max_n_sequences $N --embed_idx $LAST_IDX --k 4096 --approximate $CONFIG_ARGS
    echo "inference $split: $(( $(date +%s) - t0 ))s"
done

# duplication breakdown: cc-news computes it here; Pile domains use slurm/ngram_overlap.slurm's output if present
REPORT_ARGS=""
if [ "$SUBSET" = "cc-news" ]; then
    if [ ! -f data/cc-news/test_13gram_overlap.npy ]; then
        python3 -I "$REPO/scripts/prepare_cc_news.py" --overlap_only --out_dir data/cc-news
    fi
    REPORT_ARGS="--overlap_dir data/cc-news"
elif [ -f "data/the-pile/$SUBSET-overlap/test_13gram_overlap.npy" ]; then
    REPORT_ARGS="--overlap_dir data/the-pile/$SUBSET-overlap"
fi
PYTHONPATH=$REPO python3 "$REPO/scripts/report_knn.py" --subset "$SUBSET" --embed_idx $LAST_IDX \
    --max_n_sequences $N --k 4096 --approximate $REPORT_ARGS
echo "eval done"
