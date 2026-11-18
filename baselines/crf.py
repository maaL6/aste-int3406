"""Baseline CRF + hồi quy logistic: học máy với đặc trưng thủ công, không dùng học sâu.

Ba bước nối tiếp, mỗi bước một mô hình, huấn luyện riêng cho từng tập dữ liệu:
  1. CRF gán nhãn BIO cho khía cạnh và ý kiến (đặc trưng: từ, hậu tố, từ loại, quan hệ phụ thuộc,
     từ cha, từ điển cảm xúc, từ lân cận).
  2. Hồi quy logistic quyết định khía cạnh nào đi với ý kiến nào (khoảng cách, đường đi trên cây cú
     pháp, từ loại, số span xen giữa).
  3. Hồi quy logistic phân loại cảm xúc của cặp (từ ý kiến, từ điển, phủ định, từ khía cạnh).

Bước 2 và 3 học trên span đúng của tập train; khi dự đoán thì dùng span do CRF trích ra.
Siêu tham số cố định, không dò. Cả ba mô hình đều tối ưu bằng L-BFGS nên không có tính ngẫu nhiên:
chạy lại cho đúng cùng kết quả, vì vậy chỉ có `seed0`.

Cần: pip install -r baselines/requirements.txt
Chạy từ thư mục gốc:  python baselines/crf.py
"""

import json
import os
import time

import sklearn_crfsuite
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from common import DATASETS, ROOT, children, is_negated, load, load_lexicon, span_head, tree_distance
from aste_eval import count_index, evaluate_file, make_record, prf, write_preds


# ---------------------------------------------------------------- bước 1: CRF

def word_features(sent, i, lexicon, prefix=""):
    word = sent["tokens"][i]
    lower = word.lower()
    return {
        prefix + "word": lower,
        prefix + "pos": sent["pos"][i],
        prefix + "pos2": sent["pos"][i][:2],
        prefix + "lexicon": lexicon.get(lower, "none"),
    }


def token_features(sent, i, lexicon):
    word = sent["tokens"][i]
    lower = word.lower()
    head = sent["head"][i]
    feats = {
        "bias": 1.0,
        "suffix3": lower[-3:],
        "suffix2": lower[-2:],
        "is_title": word.istitle(),
        "is_upper": word.isupper(),
        "is_digit": word.isdigit(),
        "deprel": sent["deprel"][i],
        "head_pos": sent["pos"][head][:2] if head >= 0 else "ROOT",
        "head_word": sent["tokens"][head].lower() if head >= 0 else "ROOT",
        "head_lexicon": lexicon.get(sent["tokens"][head].lower(), "none") if head >= 0 else "none",
    }
    feats.update(word_features(sent, i, lexicon))
    for offset in (-2, -1, 1, 2):
        j = i + offset
        if 0 <= j < len(sent["tokens"]):
            feats.update(word_features(sent, j, lexicon, prefix="{:+d}:".format(offset)))
        else:
            feats["{:+d}:pad".format(offset)] = 1.0
    return feats


def bio_tags(sent):
    tags = ["O"] * len(sent["tokens"])
    for a_start, a_end, o_start, o_end, _ in sent["triplets"]:
        for start, end, kind in ((a_start, a_end, "A"), (o_start, o_end, "P")):
            tags[start] = "B-" + kind
            for i in range(start + 1, end + 1):
                tags[i] = "I-" + kind
    return tags


def spans_from_tags(tags, kind):
    spans, start = [], None
    for i, tag in enumerate(list(tags) + ["O"]):
        if start is not None and tag != "I-" + kind:
            spans.append((start, i - 1))
            start = None
        if tag == "B-" + kind:
            start = i
    return spans


# ---------------------------------------------------------------- bước 2 và 3: đặc trưng của cặp

def pair_features(sent, aspect, opinion, aspects, opinions):
    a, o = span_head(sent, *aspect), span_head(sent, *opinion)
    distance = min(abs(aspect[0] - opinion[1]), abs(opinion[0] - aspect[1]))
    left, right = min(aspect[1], opinion[1]), max(aspect[0], opinion[0])
    between = sent["tokens"][left + 1:right]
    tree = tree_distance(sent, a, o)
    return {
        "bias": 1.0,
        "distance": min(distance, 10) / 10.0,
        "distance=" + str(min(distance, 6)): 1.0,
        "tree=" + str(min(tree, 5)): 1.0,
        "aspect_first": aspect[0] < opinion[0],
        "direct": sent["head"][o] == a or sent["head"][a] == o,
        "same_head": sent["head"][o] == sent["head"][a],
        "aspect_deprel": sent["deprel"][a],
        "opinion_deprel": sent["deprel"][o],
        "deprels": sent["deprel"][a] + "|" + sent["deprel"][o],
        "opinion_pos": sent["pos"][o][:2],
        "aspects_between": sum(1 for s in aspects if left < s[0] and s[1] < right),
        "opinions_between": sum(1 for s in opinions if left < s[0] and s[1] < right),
        "comma_between": "," in between,
        "and_between": "and" in between or "but" in between,
        "closest_aspect": all(abs(span_head(sent, *s) - o) >= abs(a - o) for s in aspects),
        "closest_opinion": all(abs(span_head(sent, *s) - a) >= abs(a - o) for s in opinions),
        "n_aspects": min(len(aspects), 4),
        "n_opinions": min(len(opinions), 4),
    }


def sentiment_features(sent, aspect, opinion, lexicon, kids):
    a, o = span_head(sent, *aspect), span_head(sent, *opinion)
    feats = {"bias": 1.0, "aspect_head": sent["tokens"][a].lower()}
    negated = False
    for i in range(opinion[0], opinion[1] + 1):
        lower = sent["tokens"][i].lower()
        feats["opinion_word=" + lower] = 1.0
        feats["opinion_lexicon=" + lexicon.get(lower, "none")] = 1.0
        negated = negated or is_negated(sent, i, kids)
    if sent["head"][o] >= 0:
        negated = negated or is_negated(sent, sent["head"][o], kids)
    feats["negated"] = negated
    feats["head_lexicon+negated"] = lexicon.get(sent["tokens"][o].lower(), "none") + "|" + str(negated)
    for i in (opinion[0] - 1, opinion[0] - 2):
        if i >= 0:
            feats["before_opinion=" + sent["tokens"][i].lower()] = 1.0
    return feats


# ---------------------------------------------------------------- huấn luyện và dự đoán

class Pipeline(object):
    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=200,
                                        all_possible_transitions=True)
        self.pair_model = make_pipeline(DictVectorizer(), LogisticRegression(C=1.0, max_iter=2000))
        self.sentiment_model = make_pipeline(DictVectorizer(), LogisticRegression(C=1.0, max_iter=2000))

    def fit(self, sents):
        self.crf.fit([[token_features(s, i, self.lexicon) for i in range(len(s["tokens"]))] for s in sents],
                     [bio_tags(s) for s in sents])
        pair_x, pair_y, sent_x, sent_y = [], [], [], []
        for s in sents:
            kids = children(s)
            gold = {(t[0], t[1], t[2], t[3]): t[4] for t in s["triplets"]}
            aspects = sorted(set((t[0], t[1]) for t in s["triplets"]))
            opinions = sorted(set((t[2], t[3]) for t in s["triplets"]))
            for aspect in aspects:
                for opinion in opinions:
                    pair_x.append(pair_features(s, aspect, opinion, aspects, opinions))
                    pair_y.append(aspect + opinion in gold)
                    if aspect + opinion in gold:
                        sent_x.append(sentiment_features(s, aspect, opinion, self.lexicon, kids))
                        sent_y.append(gold[aspect + opinion])
        self.pair_model.fit(pair_x, pair_y)
        self.sentiment_model.fit(sent_x, sent_y)
        return self

    def predict(self, sent):
        tags = self.crf.predict_single([token_features(sent, i, self.lexicon)
                                        for i in range(len(sent["tokens"]))])
        aspects, opinions = spans_from_tags(tags, "A"), spans_from_tags(tags, "P")
        if not aspects or not opinions:
            return []
        kids = children(sent)
        candidates = [(a, o) for a in aspects for o in opinions]
        keep = self.pair_model.predict([pair_features(sent, a, o, aspects, opinions) for a, o in candidates])
        pairs = [c for c, k in zip(candidates, keep) if k]
        if not pairs:
            return []
        sentiments = self.sentiment_model.predict(
            [sentiment_features(sent, a, o, self.lexicon, kids) for a, o in pairs])
        return [a + o + (s,) for (a, o), s in zip(pairs, sentiments)]

    def n_parameters(self):
        n = len(self.crf.state_features_) + len(self.crf.transition_features_)
        for model in (self.pair_model, self.sentiment_model):
            lr = model.steps[-1][1]
            n += lr.coef_.size + lr.intercept_.size
        return n


def index_f1(sents, preds):
    return prf(*count_index(sents, [make_record(s, p) for s, p in zip(sents, preds)]))[2]


def main():
    lexicon = load_lexicon()
    for tap in DATASETS:
        train, dev, test = load(tap, "train"), load(tap, "dev"), load(tap, "test")
        start = time.time()
        model = Pipeline(lexicon).fit(train)
        train_minutes = (time.time() - start) / 60
        dev_f1 = index_f1(dev, [model.predict(s) for s in dev])
        start = time.time()
        preds = [model.predict(s) for s in test]
        ms = 1000 * (time.time() - start) / len(test)

        path = os.path.join(ROOT, "results", "preds", "crf", tap, "seed0.jsonl")
        write_preds(path, tap, preds)
        res = evaluate_file(path)
        stats = {"dev_f1": dev_f1, "phut_train": train_minutes, "ms_moi_cau": ms,
                 "trieu_tham_so": model.n_parameters() / 1e6}
        log_dir = os.path.join(ROOT, "results", "logs", "crf", tap)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "seed0.stats.json"), "w") as f:
            json.dump(stats, f, indent=1)
        print("{}  dev F1 {:.2f}  test P {:.2f} R {:.2f} F1 {:.2f}  (train {:.2f} phút, {:.2f} ms/câu, {:.3f} triệu tham số)".format(
            tap, dev_f1, res["test_p_chi_so"], res["test_r_chi_so"], res["test_f1_chi_so"],
            train_minutes, ms, stats["trieu_tham_so"]))


if __name__ == "__main__":
    main()
