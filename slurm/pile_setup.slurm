#!/bin/bash
#SBATCH --job-name=pile-setup
#SBATCH --account=bcyi-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --time=03:00:00
#SBATCH --output=/work/hdd/bcyi/eergun/silo-knn/logs/%x-%j.out
#
# Data and tokenization for the kNN-LM datastore of a large Pile subset (Wikipedia_(en), Github).
# Reads the first N_FILES arrow files of the subset (~0.5 GB each) and samples a pool of
# TARGET_TOKENS tokens from them; build_dstore.slurm then encodes random 10K-window shards of the pool.
# Reuses the python env and the Pile val/test files made by slurm/enron_setup.slurm. Every step
# is skipped when its output already exists, so the job can be resubmitted.
#
#   sbatch --job-name=wiki-setup --export=ALL,SUBSET='Wikipedia_(en)' slurm/pile_setup.slurm
#   sbatch --job-name=github-setup --export=ALL,SUBSET=Github slurm/pile_setup.slurm
set -eo pipefail   # no -u: module and venv activation scripts reference unset variables

case "$SUBSET" in
    "Wikipedia_(en)") PILE_SET="Wikipedia (en)"; N_FILES=${N_FILES:-8} ;;
    Github)           PILE_SET="Github";         N_FILES=${N_FILES:-4} ;;
    *) echo "unknown SUBSET='$SUBSET'"; exit 1 ;;
esac
TARGET_TOKENS=${TARGET_TOKENS:-400000000}   # about the size of cc-news; ~78 shards at most

REPO=/work/nvme/bcyi/eergun/silo-lm
WORK=/work/hdd/bcyi/eergun/silo-knn       # data/ and out/ are resolved relative to this directory
RAW="$WORK/raw/armelr_$SUBSET"            # untrusted downloads, kept in their own directory
VENV=$WORK/venv
HF_DS=https://huggingface.co/datasets

module load pytorch-conda
source "$VENV/bin/activate"
mkdir -p "$WORK"/{data/the-pile,out} "$RAW/train"

fetch() {  # fetch <url> <dest>: download to a temp file, then move into place
    [ -f "$2" ] && return 0
    curl -fsSL --retry 5 --retry-delay 5 -o "$2.part" "$1" && mv "$2.part" "$2"
}

echo "== [1/4] download the first $N_FILES files of $PILE_SET"
N_TOTAL=$(curl -fsSL "$HF_DS/ArmelR/the-pile-splitted/resolve/main/data/${PILE_SET// /%20}/train/state.json" \
          | sed -n 's/.*"data-00000-of-\([0-9]*\)\.arrow".*/\1/p' | head -1)
[[ "$N_TOTAL" =~ ^[0-9]{5}$ ]] || { echo "could not read the number of files: '$N_TOTAL'"; exit 1; }
for i in $(seq 0 $((N_FILES - 1))); do
    f=$(printf "data-%05d-of-%s.arrow" $i "$N_TOTAL")
    fetch "$HF_DS/ArmelR/the-pile-splitted/resolve/main/data/${PILE_SET// /%20}/train/$f" "$RAW/train/$f"
done
for split in val test; do
    fetch "$HF_DS/monology/pile-uncopyrighted/resolve/main/$split.jsonl.zst" "$WORK/data/the-pile/$split.jsonl.zst"
done
du -sh "$RAW"

echo "== [2/4] dedup, filter against val/test, sample $TARGET_TOKENS tokens"
TRAIN_FILE="$WORK/data/the-pile/train-gz/$SUBSET-00.json.gz"
if [ ! -f "$TRAIN_FILE.done" ]; then
    python3 -I "$REPO/scripts/prepare_pile_subset.py" --pile_set_name "$PILE_SET" \
        --raw_dir "$RAW" --splits train --n_files "$N_FILES" --target_tokens "$TARGET_TOKENS" \
        --pile_dir "$WORK/data/the-pile" --out_path "$TRAIN_FILE"
fi
cat "$TRAIN_FILE.stats.json"

echo "== [3/4] tokenize"
cd "$WORK"
for split in train val test; do
    PYTHONPATH=$REPO python3 "$REPO/scripts/main.py" --task tokenize --split $split \
        --subset "$SUBSET" --lm pythia-1.4B
done

echo "== [4/4] number of datastore shards (10K windows each)"
SUBSET="$SUBSET" python3 - <<'EOF'
import os, pickle as pkl, math
subset = os.environ["SUBSET"]
path = "out/neoX/train/%s-tokenized.pkl" % subset
n = sum(len(ids) for ids in pkl.load(open(path, "rb")))
# same window arithmetic as model.batch_merged (max_seq_length=1024, stride=512)
n_windows = 0
for begin in range(0, n - 1, 512):
    n_windows += 1
    if min(begin + 1024, n - 1) == n - 1:
        break
n_shards = math.ceil(n_windows / 10000)
print("tokens=%d windows=%d shards=%d (SILO kNN-LM datastore: 1024.0M tokens)" % (n, n_windows, n_shards))
open("out/neoX/train/%s-nshards.txt" % subset, "w").write("%d\n" % n_shards)
EOF
echo "setup done"
