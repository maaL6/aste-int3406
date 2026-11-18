"""Baseline luật cú pháp + từ điển cảm xúc, không huấn luyện.

Ý tưởng theo Double Propagation (Qiu et al., Computational Linguistics 2011): từ ý kiến và khía cạnh
được trích cùng lúc nhờ quan hệ cú pháp giữa chúng. Từ ý kiến ban đầu lấy từ từ điển của Hu & Liu
(KDD 2004). Cảm xúc lấy từ từ điển, đảo dấu khi có từ phủ định.

Các luật (O là từ ý kiến, T là danh từ làm khía cạnh):
  1. amod:   O bổ nghĩa trực tiếp cho T                 ("great screen")
  2. nsubj:  T là chủ ngữ của O                         ("the screen is great")
  3. obj:    T là tân ngữ của động từ ý kiến O          ("I love the screen")
  4. xcomp:  T là chủ ngữ của động từ V, O phụ thuộc V  ("the screen looks great", "it works well")
  5. conj:   O2 liên hợp với O1 thì dùng chung khía cạnh; T2 liên hợp với T1 thì dùng chung ý kiến
Khía cạnh được mở rộng sang trái theo quan hệ compound ("battery life").

Tập train chỉ dùng để thống kê: cảm xúc hay gặp nhất của từ ý kiến không có trong từ điển.
Tập dev dùng để chọn bộ luật (xem `python baselines/rule.py --tune`).

Chạy từ thư mục gốc:  python baselines/rule.py
"""

import argparse
import os
import time
from collections import Counter, defaultdict

from common import DATASETS, ROOT, children, is_negated, load, load_lexicon
from aste_eval import count_index, evaluate_file, make_record, prf, write_preds

# Bộ luật đã chọn trên dev (trung bình F1 của 4 tập), xem --tune.
DEFAULT_CONFIG = {
    "rules": ("amod", "nsubj", "obj", "xcomp", "conj"),
    "opinion_pos": ("JJ", "VB", "RB"),   # từ loại được nhận làm từ ý kiến
    "any_adjective": False,              # nhận mọi tính từ làm từ ý kiến, kể cả ngoài từ điển
    "expand_compound": True,
}

FLIP = {"POS": "NEG", "NEG": "POS", "NEU": "NEU"}


def train_sentiment_stats(train_sents):
    """Cảm xúc hay gặp nhất của mỗi từ ý kiến (một từ) trong tập train."""
    counts = defaultdict(Counter)
    for sent in train_sents:
        for a_start, a_end, o_start, o_end, sentiment in sent["triplets"]:
            if o_start == o_end:
                counts[sent["tokens"][o_start].lower()][sentiment] += 1
    return {word: c.most_common(1)[0][0] for word, c in counts.items()}


def is_opinion(sent, i, lexicon, config):
    pos = sent["pos"][i][:2]
    if pos not in config["opinion_pos"]:
        return False
    if sent["tokens"][i].lower() in lexicon:
        return True
    return config["any_adjective"] and pos == "JJ"


def is_noun(sent, i):
    return sent["pos"][i].startswith("NN")


def aspect_span(sent, i, kids, config):
    start = i
    if config["expand_compound"]:
        while start - 1 in kids[i] and sent["deprel"][start - 1] == "compound":
            start -= 1
    return start, i


def extract_pairs(sent, lexicon, config):
    """Trả về tập các cặp (vị trí từ khía cạnh, vị trí từ ý kiến)."""
    n = len(sent["tokens"])
    head, deprel = sent["head"], sent["deprel"]
    rules = config["rules"]
    opinions = [i for i in range(n) if is_opinion(sent, i, lexicon, config)]
    pairs = set()

    for o in opinions:
        h = head[o]
        if "amod" in rules and deprel[o] == "amod" and h >= 0 and is_noun(sent, h):
            pairs.add((h, o))
        for t in range(n):
            if head[t] != o or not is_noun(sent, t):
                continue
            if "nsubj" in rules and deprel[t] in ("nsubj", "nsubj:pass"):
                pairs.add((t, o))
            if "obj" in rules and deprel[t] == "obj" and sent["pos"][o].startswith("VB"):
                pairs.add((t, o))
        if "xcomp" in rules and deprel[o] in ("xcomp", "advmod") and h >= 0 and sent["pos"][h].startswith("VB"):
            for t in range(n):
                if head[t] == h and deprel[t] in ("nsubj", "nsubj:pass") and is_noun(sent, t):
                    pairs.add((t, o))

    if "conj" in rules:
        for _ in range(2):
            for t, o in list(pairs):
                for i in range(n):
                    if deprel[i] != "conj":
                        continue
                    if head[i] == o and i in opinions:
                        pairs.add((t, i))
                    if head[i] == t and is_noun(sent, i):
                        pairs.add((i, o))
    return pairs


def predict(sent, lexicon, train_sentiment, config=DEFAULT_CONFIG):
    kids = children(sent)
    triplets = set()
    for t, o in extract_pairs(sent, lexicon, config):
        word = sent["tokens"][o].lower()
        sentiment = lexicon.get(word) or train_sentiment.get(word, "POS")
        negated = is_negated(sent, o, kids)
        # "is not good": từ phủ định có thể gắn vào động từ mà từ ý kiến phụ thuộc
        if not negated and sent["deprel"][o] in ("xcomp", "advmod") and sent["head"][o] >= 0:
            negated = is_negated(sent, sent["head"][o], kids)
        if negated:
            sentiment = FLIP[sentiment]
        a_start, a_end = aspect_span(sent, t, kids, config)
        triplets.add((a_start, a_end, o, o, sentiment))
    return sorted(triplets)


def score(tap, split, lexicon, config):
    train_sentiment = train_sentiment_stats(load(tap, "train"))
    sents = load(tap, split)
    records = [make_record(s, predict(s, lexicon, train_sentiment, config)) for s in sents]
    return prf(*count_index(sents, records))


def tune(lexicon):
    """Thử lần lượt từng thay đổi so với bộ luật mặc định, chấm trên dev."""
    variants = [("mặc định", {})]
    for rule in DEFAULT_CONFIG["rules"]:
        variants.append(("bỏ luật " + rule, {"rules": tuple(r for r in DEFAULT_CONFIG["rules"] if r != rule)}))
    variants += [
        ("từ ý kiến: chỉ tính từ", {"opinion_pos": ("JJ",)}),
        ("từ ý kiến: tính từ, động từ", {"opinion_pos": ("JJ", "VB")}),
        ("từ ý kiến: thêm danh từ", {"opinion_pos": ("JJ", "VB", "RB", "NN")}),
        ("nhận mọi tính từ", {"any_adjective": True}),
        ("không mở rộng compound", {"expand_compound": False}),
    ]
    print("F1 trên dev (so trên chỉ số)")
    print("{:<30}".format("") + "".join("{:>8}".format(t) for t in DATASETS) + "{:>8}".format("TB"))
    for name, change in variants:
        config = dict(DEFAULT_CONFIG, **change)
        f1s = [score(tap, "dev", lexicon, config)[2] for tap in DATASETS]
        print("{:<30}".format(name) + "".join("{:>8.2f}".format(f) for f in f1s)
              + "{:>8.2f}".format(sum(f1s) / len(f1s)))


def main():
    parser = argparse.ArgumentParser(description="Baseline luật cú pháp + từ điển cho ASTE.")
    parser.add_argument("--tune", action="store_true", help="so sánh các biến thể của bộ luật trên dev")
    args = parser.parse_args()
    lexicon = load_lexicon()
    if args.tune:
        tune(lexicon)
        return
    for tap in DATASETS:
        train_sentiment = train_sentiment_stats(load(tap, "train"))
        test = load(tap, "test")
        start = time.time()
        preds = [predict(s, lexicon, train_sentiment) for s in test]
        ms = 1000 * (time.time() - start) / len(test)
        path = os.path.join(ROOT, "results", "preds", "rule", tap, "seed0.jsonl")
        write_preds(path, tap, preds)
        res = evaluate_file(path)
        print("{}  dev F1 {:.2f}  test P {:.2f} R {:.2f} F1 {:.2f}  ({:.3f} ms/câu)".format(
            tap, score(tap, "dev", lexicon, DEFAULT_CONFIG)[2],
            res["test_p_chi_so"], res["test_r_chi_so"], res["test_f1_chi_so"], ms))


if __name__ == "__main__":
    main()
