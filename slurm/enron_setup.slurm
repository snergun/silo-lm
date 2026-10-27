#!/bin/bash
#SBATCH --job-name=enron-setup
#SBATCH --account=bcyi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bcyi/eergun/silo-knn/logs/%x-%j.out
#
# Environment, data and tokenization for the Enron_Emails kNN-LM datastore.
# Every step is skipped when its output already exists, so the job can be resubmitted.
set -eo pipefail   # no -u: module and venv activation scripts reference unset variables

REPO=/work/nvme/bcyi/eergun/silo-lm
WORK=/work/hdd/bcyi/eergun/silo-knn     # data/ and out/ are resolved relative to this directory
RAW=$WORK/raw/armelr_enron              # untrusted downloads, kept in their own directory
VENV=$WORK/venv
HF_DS=https://huggingface.co/datasets

module load pytorch-conda
mkdir -p "$WORK"/{data/the-pile,out,logs} "$RAW"/{train,test}

echo "== [1/5] python env with faiss"
if [ ! -f "$VENV/.done" ]; then
    python3 -m venv --system-site-packages "$VENV"
    source "$VENV/bin/activate"
    echo "numpy==$(python3 -c 'import numpy; print(numpy.__version__)')" > "$WORK/constraints.txt"
    pip install --no-cache-dir -c "$WORK/constraints.txt" faiss-gpu-cu12 \
        || pip install --no-cache-dir -c "$WORK/constraints.txt" faiss-cpu
    python3 -c 'import faiss; print("faiss", faiss.__version__, "gpu build:", hasattr(faiss, "StandardGpuResources"))'
    touch "$VENV/.done"
else
    source "$VENV/bin/activate"
fi

fetch() {  # fetch <url> <dest>: download to a temp file, then move into place
    [ -f "$2" ] && return 0
    curl -fsSL --retry 5 --retry-delay 5 -o "$2.part" "$1" && mv "$2.part" "$2"
}

echo "== [2/5] downloads"
for split in train test; do
    for i in $(seq -f "%05g" 0 95); do
        fetch "$HF_DS/ArmelR/the-pile-splitted/resolve/main/data/Enron%20Emails/$split/data-$i-of-00096.arrow" \
              "$RAW/$split/data-$i-of-00096.arrow"
    done
done
for split in val test; do
    fetch "$HF_DS/monology/pile-uncopyrighted/resolve/main/$split.jsonl.zst" "$WORK/data/the-pile/$split.jsonl.zst"
done
huggingface-cli download EleutherAI/pythia-1.4b-deduped-v0 --exclude "pytorch_model.bin" > /dev/null
du -sh "$RAW" "$WORK/data/the-pile"

echo "== [3/5] dedup, filter against val/test, sample ~45M tokens"
TRAIN_FILE=$WORK/data/the-pile/train-gz/Enron_Emails-00.json.gz
if [ ! -f "$TRAIN_FILE.done" ]; then
    python3 -I "$REPO/scripts/prepare_pile_subset.py" \
        --raw_dir "$RAW" --pile_dir "$WORK/data/the-pile" --out_path "$TRAIN_FILE"
fi
cat "$TRAIN_FILE.stats.json"

echo "== [4/5] tokenize"
cd "$WORK"
for split in train val test; do
    PYTHONPATH=$REPO python3 "$REPO/scripts/main.py" --task tokenize --split $split \
        --subset Enron_Emails --lm pythia-1.4B
done

echo "== [5/5] number of datastore shards (10K windows each)"
python3 - <<'EOF'
import pickle as pkl, math
path = "out/neoX/train/Enron_Emails-tokenized.pkl"
n = sum(len(ids) for ids in pkl.load(open(path, "rb")))
# same window arithmetic as model.batch_merged (max_seq_length=1024, stride=512)
n_windows = 0
for begin in range(0, n - 1, 512):
    n_windows += 1
    if min(begin + 1024, n - 1) == n - 1:
        break
n_shards = math.ceil(n_windows / 10000)
print("tokens=%d windows=%d shards=%d" % (n, n_windows, n_shards))
open("out/neoX/train/Enron_Emails-nshards.txt", "w").write("%d\n" % n_shards)
EOF
echo "setup done"
