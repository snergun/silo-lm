"""Show what the 13-gram overlap between a domain's val/test documents and its training data
consists of: exact copies, near-copies, quoting, or many unrelated shared phrases.

For every val/test document with >= 50% of its word 13-grams in train (the statistic in
duplicate_stats.json, same 13-gram definition as prepare_cc_news.py), finds the training documents
those 13-grams occur in and classifies the document by how the best-matching training document
relates to it. Writes a summary plus sampled examples, with the words not found in train marked,
to {out_dir}/overlap_examples.md and the per-document numbers to {out_dir}/overlap_examples.json.

Run from the directory that holds data/, inside a job.
"""
import os
import re
import json
import argparse
from collections import Counter

import numpy as np

from data import load_pile
from prepare_cc_news import ngram_hashes, NGRAM

THRESHOLD = 0.5      # the duplicate_stats.json statistic being explained
COVER = 0.9          # a document "covers" another if it holds >= 90% of the other's 13-grams
MANY = 100           # a 13-gram found in more than this many training documents counts as boilerplate
MAX_SOURCES = 10
CLASSES = [
    "exact duplicate",
    "near-duplicate",                 # one train doc holds >= 90% of it, and it holds >= 90% of that doc
    "contained in a longer train doc", # one train doc holds >= 90% of it, but has much more text
    "contains a whole train doc",     # it holds >= 90% of one train doc, plus text of its own
    "partial overlap with one train doc",
    "spread over several train docs",
]

def contains(sorted_a, x):
    if len(sorted_a) == 0:
        return np.zeros(len(x), dtype=bool)
    pos = np.minimum(np.searchsorted(sorted_a, x), len(sorted_a) - 1)
    return sorted_a[pos] == x

def collect_train_hits(train, query):
    """(13-gram hash, train doc index) for every distinct query 13-gram in every training doc, sorted by hash."""
    hashes, docs = [], []
    for i, text in enumerate(train):
        h = ngram_hashes(text)
        h = np.unique(h[contains(query, h)])
        if len(h):
            hashes.append(h)
            docs.append(np.full(len(h), i, dtype=np.int32))
    hashes, docs = np.concatenate(hashes), np.concatenate(docs)
    order = np.argsort(hashes, kind="stable")
    return hashes[order], docs[order]

def word_coverage(matched, n_words):
    """Word i is covered if it lies in some matched 13-gram (matched[s] for 13-grams starting at s)."""
    delta = np.zeros(n_words + 1, dtype=np.int64)
    starts = np.where(matched)[0]
    np.add.at(delta, starts, 1)
    np.add.at(delta, starts + NGRAM, -1)
    return np.cumsum(delta)[:n_words] > 0

def analyze(h, hit_hash, hit_doc):
    uh, inv = np.unique(h, return_inverse=True)
    weight = np.bincount(inv, minlength=len(uh))   # occurrences of each distinct 13-gram in the doc
    lo, hi = np.searchsorted(hit_hash, uh, "left"), np.searchsorted(hit_hash, uh, "right")
    n_docs_per = hi - lo
    found = n_docs_per > 0
    n_matched = int(weight[found].sum())
    r = {"n_13grams": len(h), "overlap": n_matched / len(h), "found_positions": found[inv]}
    if r["overlap"] < THRESHOLD:
        return r
    r["boilerplate_share"] = float(weight[n_docs_per > MANY].sum() / n_matched)
    pair_u = np.repeat(np.arange(len(uh)), n_docs_per)
    pair_doc = np.concatenate([hit_doc[a:b] for a, b in zip(lo[found], hi[found])])
    cand, pair_c = np.unique(pair_doc, return_inverse=True)
    # greedy cover of the matched 13-grams by training documents
    uncovered, sources, covered = found.copy(), [], 0
    while covered < COVER * n_matched and len(sources) < MAX_SOURCES:
        live = uncovered[pair_u]
        gain = np.bincount(pair_c[live], weights=weight[pair_u[live]], minlength=len(cand))
        c = int(gain.argmax())
        if not sources:
            r["best_cover"] = float(gain[c] / len(h))
            r["n_train_copies"] = int(np.sum(gain >= COVER * len(h)))
        sources.append(int(cand[c]))
        covered += gain[c]
        uncovered[pair_u[pair_c == c]] = False
    r["best_train_doc"] = sources[0]
    r["n_sources_for_90pct"] = len(sources) if covered >= COVER * n_matched else None
    r["n_train_docs_sharing_any"] = len(cand)
    r["uh"] = uh
    return r

def classify(r):
    if r["exact"]:
        return CLASSES[0]
    if r["best_cover"] >= COVER:
        return CLASSES[1] if r["reverse_cover"] >= COVER else CLASSES[2]
    if r["reverse_cover"] >= COVER and r["best_cover"] >= 0.5 * r["overlap"]:
        return CLASSES[3]
    if r["best_cover"] >= 0.8 * r["overlap"]:
        return CLASSES[4]
    return CLASSES[5]

def render(text, covered, max_chars):
    """The text with runs of words not covered by shared 13-grams wrapped in «»."""
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    if len(spans) != len(covered):   # \S+ and str.split disagree on some exotic whitespace
        words = text.split()
        text, pos = " ".join(words), 0
        spans = []
        for w in words:
            spans.append((pos, pos + len(w)))
            pos += len(w) + 1
    out, i, end = [], 0, 0
    while i < len(spans) and spans[i][0] < max_chars:
        j = i
        while j + 1 < len(spans) and covered[j + 1] == covered[i]:
            j += 1
        out.append(text[end:spans[i][0]])
        run = text[spans[i][0]:spans[j][1]]
        out.append(run if covered[i] else "«" + run + "»")
        end = spans[j][1]
        i = j + 1
    if i < len(spans):
        out.append("\n[... {} more words]".format(len(spans) - i))
    return "".join(out).replace("````", "` ` ` `")

def main(args):
    train = load_pile("train", subset=args.subset)
    evals = {split: load_pile(split, subset=args.subset) for split in ["val", "test"]}
    stored = {split: np.load(os.path.join(args.overlap_dir, "{}_13gram_overlap.npy".format(split))) for split in evals}
    eval_hashes = {split: [ngram_hashes(t) for t in texts] for split, texts in evals.items()}
    query = np.unique(np.concatenate([h for hs in eval_hashes.values() for h in hs]))
    print ("{} train docs, {} distinct val/test 13-grams".format(len(train), len(query)), flush=True)
    hit_hash, hit_doc = collect_train_hits(train, query)
    print ("{} (13-gram, train doc) hits".format(len(hit_hash)), flush=True)
    train_index = {}
    for i, t in enumerate(train):
        train_index.setdefault(t, i)

    rows, check = [], {}
    for split, texts in evals.items():
        recomputed = np.full(len(texts), np.nan)
        for i, (text, h) in enumerate(zip(texts, eval_hashes[split])):
            if len(h) == 0:
                continue
            r = analyze(h, hit_hash, hit_doc)
            recomputed[i] = r["overlap"]
            if r["overlap"] < THRESHOLD:
                continue
            best = train[r["best_train_doc"]]
            hb = ngram_hashes(best)
            r.update(split=split, doc=i, n_words=len(text.split()), exact=text in train_index,
                     reverse_cover=float(np.mean(contains(r.pop("uh"), hb))),
                     best_train_doc_n_words=len(best.split()), best_matched_positions=contains(np.unique(h), hb))
            if r["exact"]:
                r["best_train_doc"] = train_index[text]
            r["class"] = classify(r)
            rows.append(r)
        valid = ~np.isnan(stored[split])
        assert np.array_equal(valid, ~np.isnan(recomputed)), split
        check[split] = float(np.max(np.abs(stored[split][valid] - recomputed[valid])))

    os.makedirs(args.out_dir, exist_ok=True)
    keep = ["split", "doc", "class", "n_words", "n_13grams", "overlap", "exact", "best_train_doc", "best_cover",
            "reverse_cover", "best_train_doc_n_words", "n_train_copies", "n_sources_for_90pct",
            "n_train_docs_sharing_any", "boilerplate_share"]
    with open(os.path.join(args.out_dir, "overlap_examples.json"), "w") as f:
        json.dump({"max_abs_diff_vs_stored": check, "docs": [{k: r[k] for k in keep} for r in rows]}, f, indent=1)

    md = ["# {}: what the >= {:.0%} 13-gram overlap with train consists of\n".format(args.subset, THRESHOLD),
          "Word 13-grams as in prepare_cc_news.py. Overlap recomputed here matches the stored per-document "
          "arrays to within {} (max abs diff).\n".format(max(check.values())),
          "Columns: *best cover* = fraction of the doc's 13-grams found in the single best-matching train doc; "
          "*reverse* = fraction of that train doc's 13-grams found in the doc; *copies* = train docs that each hold "
          ">= 90% of the doc; *boilerplate* = share of matched 13-grams that occur in > {} train docs. "
          "In the examples, text in «» is NOT found in train (in the train doc: not shared with the eval doc).\n".format(MANY)]
    md.append("| class | " + " | ".join(evals) + " | all |\n|---|" + "---|" * (len(evals) + 1))
    for c in CLASSES:
        counts = [sum(r["class"] == c and r["split"] == s for r in rows) for s in evals]
        md.append("| {} | ".format(c) + " | ".join(map(str, counts)) + " | {} |".format(sum(counts)))
    md.append("| **total >= {:.0%}** | ".format(THRESHOLD)
              + " | ".join(str(sum(r["split"] == s for r in rows)) for s in evals) + " | {} |".format(len(rows)))
    md.append("| docs with 13-grams | " + " | ".join(str(int(np.sum(~np.isnan(stored[s])))) for s in evals) + " | |\n")
    bins = Counter()
    for r in rows:
        n = r["n_train_copies"]
        bins["0" if n == 0 else "1" if n == 1 else "2-5" if n <= 5 else "6-20" if n <= 20 else ">20"] += 1
    md.append("Train docs that each hold >= 90% of the doc (over all {} docs): ".format(len(rows))
              + ", ".join("{}: {}".format(k, bins[k]) for k in ["0", "1", "2-5", "6-20", ">20"]) + "\n")
    bp = np.array([r["boilerplate_share"] for r in rows])
    md.append("Boilerplate share of matched 13-grams: median {:.2f}; docs with share > 0.5: {}\n".format(
        np.median(bp), int(np.sum(bp > 0.5))))

    rng = np.random.default_rng(args.seed)
    for c in CLASSES:
        members = [r for r in rows if r["class"] == c]
        if not members:
            continue
        md.append("## {} ({} docs)\n".format(c, len(members)))
        for k in sorted(rng.choice(len(members), size=min(args.n_examples, len(members)), replace=False)):
            r = members[k]
            text, best = evals[r["split"]][r["doc"]], train[r["best_train_doc"]]
            md.append("### {} doc {}: {} words, overlap {:.2f}, best cover {:.2f}, reverse {:.2f}, copies {}, "
                      "sources for 90% {}, boilerplate {:.2f}\n".format(
                          r["split"], r["doc"], r["n_words"], r["overlap"], r["best_cover"], r["reverse_cover"],
                          r["n_train_copies"], r["n_sources_for_90pct"] or ">{}".format(MAX_SOURCES), r["boilerplate_share"]))
            md.append("````\n" + render(text, word_coverage(r["found_positions"], r["n_words"]), args.max_chars) + "\n````\n")
            if r["exact"]:
                md.append("Identical to train doc {}.\n".format(r["best_train_doc"]))
            else:
                md.append("Best-matching train doc {} ({} words):\n".format(r["best_train_doc"], r["best_train_doc_n_words"]))
                md.append("````\n" + render(best, word_coverage(r["best_matched_positions"], len(best.split())),
                                            args.max_chars) + "\n````\n")
    with open(os.path.join(args.out_dir, "overlap_examples.md"), "w") as f:
        f.write("\n".join(md))
    print ("wrote {}".format(args.out_dir))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--overlap_dir", type=str, required=True, help="holds {val,test}_13gram_overlap.npy from ngram_overlap.py")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_examples", type=int, default=3, help="examples per class")
    parser.add_argument("--max_chars", type=int, default=1500, help="characters shown per document")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
