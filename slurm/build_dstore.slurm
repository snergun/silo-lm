#!/bin/bash
#SBATCH --job-name=build-dstore
#SBATCH --account=bcyi-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --output=/work/hdd/bcyi/eergun/silo-knn/logs/%x-%j.out
#
# Builds a kNN-LM datastore (Pythia-1.4B keys, IVFPQ index) one shard at a time:
# encode shard i -> add it to the index -> delete its keys. Peak disk is one shard of keys (~21 GB).
# Resubmitting resumes after the last shard that has a .done marker, and a job exits cleanly before
# a shard that would not finish within its time limit, so a datastore longer than one job (~16 shards)
# is built by a chain of jobs with the same name: sbatch --dependency=singleton ... (repeated).
# Requires the domain's setup job (slurm/*_setup.slurm) to have finished.
#
#   sbatch slurm/build_dstore.slurm                                    # Enron_Emails, all shards
#   sbatch --job-name=ccnews-dstore --export=ALL,SUBSET=cc-news,WORK=/work/nvme/bcyi/eergun/silo-knn-ccnews,MAX_SHARDS=10 \
#          slurm/build_dstore.slurm                                    # cc-news, first 51.2M tokens
#   for j in 1 2 3 4; do sbatch --job-name=wiki-dstore --dependency=singleton --export=ALL,SUBSET='Wikipedia_(en)',MAX_SHARDS=40 slurm/build_dstore.slurm; done
#   for j in 1 2 3 4; do sbatch --job-name=github-dstore --dependency=singleton --export=ALL,SUBSET=Github,MAX_SHARDS=40 slurm/build_dstore.slurm; done
set -eo pipefail   # no -u: module and venv activation scripts reference unset variables

REPO=/work/nvme/bcyi/eergun/silo-lm
SUBSET=${SUBSET:-Enron_Emails}
WORK=${WORK:-/work/hdd/bcyi/eergun/silo-knn}   # data/ and out/ are resolved relative to this directory
VENV=/work/hdd/bcyi/eergun/silo-knn/venv
N=10   # --max_n_sequences: thousands of 1024-token windows per shard (~5.1M tokens)

module load pytorch-conda
source "$VENV/bin/activate"
cd "$WORK"

OUT=out/pythia-1.4B/train
NSHARDS=$(cat out/neoX/train/$SUBSET-nshards.txt)
if [ -n "$MAX_SHARDS" ] && [ "$MAX_SHARDS" -lt "$NSHARDS" ]; then NSHARDS=$MAX_SHARDS; fi
COMMON="--split train --subset $SUBSET --lm pythia-1.4B --max_seq_length 1024 --stride 512 --max_n_sequences $N"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "$SUBSET: building shards 0..$((NSHARDS - 1)) in $WORK"

JOB_END=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s 2>/dev/null || true)
SHARD_SECS=3300   # longest shard so far (encode + index) was ~2700s

for i in $(seq 0 $((NSHARDS - 1))); do
    start=$((N * i)); end=$((N * (i + 1)))
    shard="$OUT/$SUBSET-1024-512-[${start}K-${end}K]"
    index="$OUT/$SUBSET-1024-512-[0K-${end}K].index"
    if [ -f "$index.done" ]; then
        echo "shard $i already done"; continue
    fi
    # stop before a shard that cannot finish in this job; a resubmitted job resumes here
    if [ -n "$JOB_END" ] && [ $((JOB_END - $(date +%s))) -lt $SHARD_SECS ]; then
        echo "stopping before shard $i: $(( (JOB_END - $(date +%s)) / 60 )) min left; resubmit to continue"
        exit 0
    fi
    # clean up after an interrupted run: an index without a marker may be half-written,
    # and a key file without a loss file was not finished by encode.
    # Exception: the previous index is only removed after this one is fully written.
    prev="$OUT/$SUBSET-1024-512-[0K-${start}K].index"
    if [ $i -gt 0 ] && [ -f "$index" ] && [ ! -f "$prev" ]; then
        touch "$index.done"; echo "shard $i already done"; continue
    fi
    rm -f "$index"
    if [ -f "$shard-embed.float16.npy" ] && [ ! -f "$shard-loss.npy" ]; then
        rm -f "$shard-embed.float16.npy"
    fi

    t0=$(date +%s)
    PYTHONPATH=$REPO python3 "$REPO/scripts/main.py" --task encode $COMMON --embed_idx $i
    t1=$(date +%s)
    PYTHONPATH=$REPO python3 "$REPO/scripts/main.py" --task inference $COMMON --embed_idx $i \
        --skip_eval --remove_embed --remove_prev_index
    t2=$(date +%s)
    touch "$index.done"
    echo "shard $i: encode $((t1 - t0))s, index $((t2 - t1))s"
    ls -lh "$index"
done
echo "datastore done: $index"
