"""Paths and loading for the LM + kNN combiner experiments.

Everything is read from the outputs of scripts/main.py (encode and inference) and slurm/build_dstore.slurm,
relative to a work directory that holds out/ and data/ (/work/hdd/bcyi/eergun/silo-knn for Enron,
/work/nvme/bcyi/eergun/silo-knn-ccnews for cc-news). ensemble/prepare.py writes a compact copy of the
neighbors to {work}/out/pythia-1.4B/ensemble/{subset}/, which is what training reads.
"""
import json
import os

import numpy as np
import torch

LM = "pythia-1.4B"
DIMENSION = 2048
POSTFIX = "1024-512"   # --max_seq_length 1024 --stride 512

class Paths(object):

    def __init__(self, work, subset, last_idx, n=10, k=4096, probe=8):
        self.work, self.subset = work, subset
        self.datastore = "{}-[0K-{}K]".format(POSTFIX, n * (last_idx + 1))
        self.dstore_target_paths = [
            os.path.join(work, "out/neoX/train/{}-tokenized_[{}K-{}K]_flatten.npy".format(subset, n * i, n * (i + 1)))
            for i in range(last_idx + 1)]
        self.result_base_fmt = os.path.join(work, "out", LM, "train", "{}-{{}}-{}-k={}-p={}_approx".format(
            subset, self.datastore, k, probe))
        self.out = os.path.join(work, "out", LM, "ensemble", subset)
        self.cache = os.path.join(self.out, "cache")
        self.embed_out = os.path.join(self.out, "embed_out.pt")
        self.prep = os.path.join(self.out, "prep.json")

    def result_base(self, split):
        return self.result_base_fmt.format(split)

    def tokenized(self, split):
        return os.path.join(self.work, "out/neoX", split, "{}-tokenized.pkl".format(self.subset))

    def targets(self, split):
        return os.path.join(self.work, "out/neoX", split, "{}-tokenized_flatten.npy".format(self.subset))

    def loss(self, split):
        return os.path.join(self.work, "out", LM, split, "{}-{}-loss.npy".format(self.subset, POSTFIX))

    def embed(self, split):
        return os.path.join(self.work, "out", LM, split, "{}-{}-embed.float16.npy".format(self.subset, POSTFIX))

    def overlap(self, split):
        """Per-document 13-gram overlap with train (scripts/ngram_overlap.py / prepare_cc_news.py), if computed."""
        for d in ["data/cc-news", "data/the-pile/{}-overlap".format(self.subset)]:
            path = os.path.join(self.work, d, "{}_13gram_overlap.npy".format(split))
            if os.path.exists(path):
                return path
        return None

    def cached(self, split, name):
        return os.path.join(self.cache, "{}_{}.npy".format(split, name))

def load_embed(path):
    """encode writes the hidden states as a raw float16 memmap without an npy header."""
    n = os.path.getsize(path) // (2 * DIMENSION)
    return np.memmap(path, dtype=np.float16, mode="r", shape=(n, DIMENSION))

def split_at_doc_boundary(doc_starts, n, ratio):
    """The document start nearest ratio * n, so no article is split between the fit and early-stopping
    slices (from pos_lm_v2's pos_lm/data/hf_data.py)."""
    cands = np.unique(np.append(np.asarray(doc_starts, dtype=np.int64), n))
    interior = cands[(cands > 0) & (cands < n)]
    if len(interior) == 0:
        return int(ratio * n)
    return int(interior[np.argmin(np.abs(interior - ratio * n))])

class Split(object):
    """One evaluation split held in CPU memory; rows are gathered per batch and moved to the GPU."""

    def __init__(self, paths, split, k):
        assert os.path.exists(os.path.join(paths.cache, split + ".done")), "run ensemble/prepare.py first"
        self.targets = torch.from_numpy(np.load(paths.cached(split, "targets")).astype(np.int64))
        self.lm_loss = torch.from_numpy(np.load(paths.cached(split, "lm_loss")))
        self.h = torch.from_numpy(np.array(load_embed(paths.embed(split))))
        self.knn_vals = torch.from_numpy(np.ascontiguousarray(np.load(paths.cached(split, "knn_vals"), mmap_mode="r")[:, :k]))
        self.knn_scores = torch.from_numpy(np.ascontiguousarray(np.load(paths.cached(split, "knn_scores"), mmap_mode="r")[:, :k]))
        self.doc_starts = np.load(paths.cached(split, "doc_starts"))
        assert len(self.targets) == len(self.h) == len(self.knn_vals) == len(self.lm_loss)

    def __len__(self):
        return len(self.targets)

    def batch(self, idx, device):
        return {"h": self.h[idx].to(device, non_blocking=True),
                "knn_vals": self.knn_vals[idx].to(device, non_blocking=True),
                "knn_scores": self.knn_scores[idx].to(device, non_blocking=True),
                "targets": self.targets[idx].to(device, non_blocking=True)}

def load_prep(paths):
    with open(paths.prep) as f:
        return json.load(f)
