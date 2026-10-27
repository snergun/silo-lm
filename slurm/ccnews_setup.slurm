#!/bin/bash
#SBATCH --job-name=ccnews-setup
#SBATCH --account=bcyi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --time=03:00:00
#SBATCH --output=/work/hdd/bcyi/eergun/silo-knn/logs/%x-%j.out
#
# Data, duplicate statistics and tokenization for the cc-news kNN-LM datastore.
# Reuses the python env made by slurm/enron_setup.slurm. Every step is skipped when its
# output already exists, so the job can be resubmitted.
set -eo pipefail   # no -u: module and venv activation scripts reference unset variables

REPO=/work/nvme/bcyi/eergun/silo-lm
WORK=/work/nvme/bcyi/eergun/silo-knn-ccnews   # data/ and out/ are resolved relative to this directory
RAW=$WORK/raw/cc_news                          # untrusted downloads, kept in their own directory
VENV=/work/hdd/bcyi/eergun/silo-knn/venv

module load pytorch-conda
source "$VENV/bin/activate"
mkdir -p "$WORK"/{data/cc-news,out} "$RAW"

echo "== [1/4] download vblagoje/cc_news"
for i in 0 1 2 3 4; do
    f=train-0000$i-of-00005.parquet
    [ -f "$RAW/$f" ] && continue
    curl -fsSL --retry 5 --retry-delay 5 -o "$RAW/$f.part" \
        "https://huggingface.co/datasets/vblagoje/cc_news/resolve/main/plain_text/$f" && mv "$RAW/$f.part" "$RAW/$f"
done
du -sh "$RAW"

echo "== [2/4] split as in download_cc_news, duplicate statistics"
if [ ! -f "$WORK/data/cc-news/.done" ]; then
    python3 -I "$REPO/scripts/prepare_cc_news.py" --raw_dir "$RAW" --out_dir "$WORK/data/cc-news"
fi
cat "$WORK/data/cc-news/duplicate_stats.json"

echo "== [3/4] tokenize"
cd "$WORK"
for split in train val test; do
    PYTHONPATH=$REPO python3 "$REPO/scripts/main.py" --task tokenize --split $split \
        --subset cc-news --lm pythia-1.4B
done

echo "== [4/4] number of datastore shards (10K windows each)"
python3 - <<'EOF'
import pickle as pkl, math
path = "out/neoX/train/cc-news-tokenized.pkl"
n = sum(len(ids) for ids in pkl.load(open(path, "rb")))
# same window arithmetic as model.batch_merged (max_seq_length=1024, stride=512)
n_windows = 0
for begin in range(0, n - 1, 512):
    n_windows += 1
    if min(begin + 1024, n - 1) == n - 1:
        break
n_shards = math.ceil(n_windows / 10000)
print("tokens=%d windows=%d shards=%d (SILO: 382.2M tokens)" % (n, n_windows, n_shards))
open("out/neoX/train/cc-news-nshards.txt", "w").write("%d\n" % n_shards)
EOF
echo "setup done"
