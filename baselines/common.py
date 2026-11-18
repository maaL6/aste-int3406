"""Phần dùng chung của hai baseline không dùng học sâu: đọc dữ liệu, từ điển cảm xúc, cây cú pháp.

Câu và bộ ba lấy từ data/ASTE-Data-V2. Từ loại và cây cú pháp phụ thuộc lấy từ bản dữ liệu đi kèm
EMC-GCN (models/EMCGCN-ASTE/data/D2), là cùng các câu đó đã được tác giả EMC-GCN phân tích sẵn,
nên baseline không cần cài thêm bộ phân tích cú pháp.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))

from aste_eval import DATASETS, read_gold  # noqa: E402

PARSED_DIR = os.path.join(ROOT, "models", "EMCGCN-ASTE", "data", "D2")
LEXICON_DIR = os.path.join(ROOT, "baselines", "lexicon")

NEGATIONS = {"not", "n't", "nt", "never", "no", "hardly", "barely", "neither", "nor", "without"}


def load(tap, split):
    """Mỗi câu: {"id", "tokens", "triplets", "pos", "head", "deprel"}.
    head[i] là vị trí (từ 0) của từ cha của từ i, hoặc -1 nếu từ i là gốc của cây."""
    gold = read_gold(tap, split)
    parsed_name = tap[2:] + tap[:2]  # 14lap -> lap14
    with open(os.path.join(PARSED_DIR, parsed_name, split + ".json"), encoding="utf-8") as f:
        parsed = json.load(f)
    assert len(parsed) == len(gold)
    for item, pack in zip(gold, parsed):
        assert pack["sentence"].split() == item["tokens"], item["id"]
        assert len(pack["head"]) == len(item["tokens"]), item["id"]
        item["pos"] = pack["postag"]
        item["head"] = [h - 1 for h in pack["head"]]
        item["deprel"] = pack["deprel"]
    return gold


def load_lexicon():
    """Từ điển cảm xúc của Hu & Liu (KDD 2004): từ -> "POS" hoặc "NEG"."""
    lexicon = {}
    for name, label in (("positive-words.txt", "POS"), ("negative-words.txt", "NEG")):
        with open(os.path.join(LEXICON_DIR, name), encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith(";"):
                    lexicon[word] = label
    return lexicon


def children(sent):
    out = [[] for _ in sent["tokens"]]
    for i, h in enumerate(sent["head"]):
        if h >= 0:
            out[h].append(i)
    return out


def is_negated(sent, i, kids):
    """Từ i có từ phủ định phụ thuộc trực tiếp vào nó, hoặc vào động từ nối/trợ động từ của nó."""
    for k in kids[i]:
        if sent["tokens"][k].lower() in NEGATIONS:
            return True
    return False


def path_to_root(sent, i):
    path = [i]
    while sent["head"][path[-1]] >= 0 and len(path) <= len(sent["tokens"]):
        path.append(sent["head"][path[-1]])
    return path


def tree_distance(sent, i, j):
    """Số cạnh trên đường đi giữa hai từ trong cây cú pháp."""
    pi, pj = path_to_root(sent, i), path_to_root(sent, j)
    depth_j = {node: d for d, node in enumerate(pj)}
    for d, node in enumerate(pi):
        if node in depth_j:
            return d + depth_j[node]
    return len(pi) + len(pj)


def span_head(sent, start, end):
    """Từ trong span có cha nằm ngoài span (từ chính của span)."""
    for i in range(end, start - 1, -1):
        if not start <= sent["head"][i] <= end:
            return i
    return end
