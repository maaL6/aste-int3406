"""Script đánh giá chung cho mọi mô hình ASTE của nhóm (xem docs/quy-uoc-chung.md, mục 4 và 5).

Dùng từ dòng lệnh, chạy từ thư mục gốc của repo:

    python eval/aste_eval.py results/preds/gts-bert/14lap/seed0.jsonl   # chấm một file
    python eval/aste_eval.py --all                                      # chấm mọi file trong results/preds

Dùng trong code để ghi file dự đoán đúng format:

    from aste_eval import write_preds
    write_preds("results/preds/emcgcn/14lap/seed0.jsonl", "14lap", preds)

với preds[i] là danh sách bộ ba (aspect_dau, aspect_cuoi, opinion_dau, opinion_cuoi, "POS") của câu ở
dòng i trong data/ASTE-Data-V2/14lap/test_triplets.txt (chỉ số từ 0, gồm cả hai đầu).

Chỉ dùng thư viện chuẩn, chạy được từ Python 3.7.
"""

import argparse
import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V2_DIR = ROOT / "data" / "ASTE-Data-V2"
PREDS_DIR = ROOT / "results" / "preds"

DATASETS = ["14lap", "14res", "15res", "16res"]
SPLITS = ["train", "dev", "test"]
SENTIMENTS = ["POS", "NEG", "NEU"]


class FormatError(Exception):
    """File dự đoán sai format hoặc không khớp với tập test."""


# ---------------------------------------------------------------- đáp án và id

def make_id(tap, split, line_no):
    return "{}-{}-{}".format(tap, split, line_no)


def read_gold(tap, split="test"):
    """Đọc đáp án gốc. Trả về danh sách theo thứ tự dòng, mỗi phần tử là
    {"id", "tokens", "triplets"}, bộ ba là (a_dau, a_cuoi, o_dau, o_cuoi, cam_xuc).

    Bộ ba bị ghi lặp trong file gốc chỉ được tính một lần (14lap test có 2 bộ ba như vậy ở dòng 281,
    nên mẫu số của recall là 541 chứ không phải 543)."""
    if tap not in DATASETS:
        raise ValueError("Tập không hợp lệ: {} (chỉ nhận {})".format(tap, DATASETS))
    if split not in SPLITS:
        raise ValueError("Phần chia không hợp lệ: {} (chỉ nhận {})".format(split, SPLITS))
    out = []
    path = V2_DIR / tap / "{}_triplets.txt".format(split)
    with open(str(path), encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            sentence, triplets_str = line.rstrip("\n").rsplit("####", 1)
            triplets = []
            for aspect, opinion, sentiment in ast.literal_eval(triplets_str):
                triplets.append((min(aspect), max(aspect), min(opinion), max(opinion), sentiment))
            out.append({
                "id": make_id(tap, split, line_no),
                "tokens": sentence.split(),
                "triplets": sorted(set(triplets)),
            })
    return out


class IdLookup(object):
    """Hàm tra `id` từ nội dung câu, dùng khi mô hình xáo thứ tự câu.

        lookup = IdLookup("14lap")
        lookup.get("Boot time is super fast , ...")   # -> "14lap-test-0"

    Vài câu xuất hiện hai lần trong cùng một tập; gọi get() lần thứ hai với cùng câu sẽ trả về id
    của lần xuất hiện thứ hai. So câu theo danh sách từ, không phân biệt hoa thường.
    """

    def __init__(self, tap, split="test"):
        self._ids = defaultdict(list)
        self._used = defaultdict(int)
        for item in read_gold(tap, split):
            self._ids[self._key(item["tokens"])].append(item["id"])

    @staticmethod
    def _key(sentence):
        tokens = sentence.split() if isinstance(sentence, str) else sentence
        return " ".join(tokens).lower()

    def get(self, sentence):
        key = self._key(sentence)
        ids = self._ids.get(key)
        if not ids:
            raise KeyError("Không tìm thấy câu trong tập: {!r}".format(key))
        n = self._used[key]
        if n >= len(ids):
            raise KeyError("Câu được tra nhiều lần hơn số lần nó xuất hiện: {!r}".format(key))
        self._used[key] += 1
        return ids[n]


# ---------------------------------------------------------------- ghi file dự đoán

def make_record(gold_item, triplets):
    """Tạo một dòng của file dự đoán từ các bộ ba theo chỉ số."""
    tokens = gold_item["tokens"]
    pred = []
    for a_start, a_end, o_start, o_end, sentiment in sorted(set(tuple(t) for t in triplets)):
        pred.append({
            "aspect": [a_start, a_end],
            "opinion": [o_start, o_end],
            "sentiment": sentiment,
            "aspect_text": " ".join(tokens[a_start:a_end + 1]),
            "opinion_text": " ".join(tokens[o_start:o_end + 1]),
        })
    return {"id": gold_item["id"], "tokens": tokens, "pred": pred}


def make_text_record(gold_item, triplets):
    """Như make_record nhưng cho mô hình sinh văn bản: bộ ba là (aspect_text, opinion_text, cam_xuc)."""
    pred = []
    seen = set()
    for aspect_text, opinion_text, sentiment in triplets:
        key = (normalize(aspect_text), normalize(opinion_text), sentiment)
        if key in seen:
            continue
        seen.add(key)
        pred.append({"aspect": None, "opinion": None, "sentiment": sentiment,
                     "aspect_text": aspect_text, "opinion_text": opinion_text})
    return {"id": gold_item["id"], "tokens": gold_item["tokens"], "pred": pred}


def write_preds(path, tap, preds, text_only=False):
    """Ghi file dự đoán cho tập test của `tap`. preds[i] là các bộ ba của câu ở dòng i.
    Kiểm tra lại file vừa ghi, báo lỗi nếu sai format."""
    gold = read_gold(tap, "test")
    if len(preds) != len(gold):
        raise FormatError("Cần {} câu, nhận được {}".format(len(gold), len(preds)))
    make = make_text_record if text_only else make_record
    path = Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        for item, triplets in zip(gold, preds):
            f.write(json.dumps(make(item, triplets), ensure_ascii=False) + "\n")
    read_preds(path, tap)


# ---------------------------------------------------------------- đọc và kiểm tra file dự đoán

def normalize(text):
    """Chuẩn hóa văn bản để so khớp: chữ thường, gộp khoảng trắng."""
    return " ".join(text.lower().split())


def _check_span(span, n_tokens, where):
    ok = (isinstance(span, list) and len(span) == 2 and all(type(x) is int for x in span)
          and 0 <= span[0] <= span[1] < n_tokens)
    if not ok:
        raise FormatError("{}: span phải là [đầu, cuối] với 0 <= đầu <= cuối < {}, nhận {!r}".format(
            where, n_tokens, span))


def read_preds(path, tap=None):
    """Đọc và kiểm tra file dự đoán. Trả về (tap, danh sách dự đoán theo đúng thứ tự tập test)."""
    records = []
    with open(str(path), encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError as e:
                raise FormatError("Dòng {}: không phải JSON ({})".format(line_no, e))
    if not records:
        raise FormatError("File rỗng")
    for i, rec in enumerate(records, 1):
        if not isinstance(rec, dict) or set(rec) != {"id", "tokens", "pred"}:
            raise FormatError("Dòng {}: cần đúng ba trường id, tokens, pred".format(i))

    if tap is None:
        tap = str(records[0]["id"]).split("-")[0]
    gold = read_gold(tap, "test")
    gold_by_id = {g["id"]: g for g in gold}

    by_id = {}
    for rec in records:
        rid = rec["id"]
        if rid not in gold_by_id:
            raise FormatError("id không có trong tập test của {}: {!r}".format(tap, rid))
        if rid in by_id:
            raise FormatError("id bị lặp: {}".format(rid))
        by_id[rid] = rec
    missing = [g["id"] for g in gold if g["id"] not in by_id]
    if missing:
        raise FormatError("Thiếu {} câu, ví dụ {}".format(len(missing), missing[:3]))

    for g in gold:
        rec = by_id[g["id"]]
        where = g["id"]
        if rec["tokens"] != g["tokens"]:
            raise FormatError("{}: tokens khác với câu gốc".format(where))
        if not isinstance(rec["pred"], list):
            raise FormatError("{}: pred phải là danh sách".format(where))
        seen = set()
        for t in rec["pred"]:
            fields = {"aspect", "opinion", "sentiment", "aspect_text", "opinion_text"}
            if not isinstance(t, dict) or set(t) != fields:
                raise FormatError("{}: mỗi bộ ba cần đúng các trường {}".format(where, sorted(fields)))
            if t["sentiment"] not in SENTIMENTS:
                raise FormatError("{}: cảm xúc phải là POS/NEG/NEU, nhận {!r}".format(where, t["sentiment"]))
            if not isinstance(t["aspect_text"], str) or not isinstance(t["opinion_text"], str):
                raise FormatError("{}: aspect_text và opinion_text phải là chuỗi".format(where))
            if (t["aspect"] is None) != (t["opinion"] is None):
                raise FormatError("{}: aspect và opinion phải cùng có chỉ số hoặc cùng null".format(where))
            if t["aspect"] is not None:
                for name in ("aspect", "opinion"):
                    _check_span(t[name], len(g["tokens"]), where)
                    expected = " ".join(g["tokens"][t[name][0]:t[name][1] + 1])
                    if t[name + "_text"] != expected:
                        raise FormatError("{}: {}_text {!r} không khớp chỉ số {} ({!r})".format(
                            where, name, t[name + "_text"], t[name], expected))
                key = (tuple(t["aspect"]), tuple(t["opinion"]), t["sentiment"])
            else:
                key = (normalize(t["aspect_text"]), normalize(t["opinion_text"]), t["sentiment"])
            if key in seen:
                raise FormatError("{}: bộ ba bị ghi trùng {}".format(where, key))
            seen.add(key)
    return tap, [by_id[g["id"]] for g in gold]


# ---------------------------------------------------------------- chấm điểm

def prf(n_correct, n_pred, n_gold):
    p = n_correct / n_pred if n_pred else 0.0
    r = n_correct / n_gold if n_gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return 100 * p, 100 * r, 100 * f1


def count_text(gold, preds):
    """So trên văn bản chuẩn hóa. Trong một câu, các bộ ba có cùng văn bản được tính là một."""
    n_correct = n_pred = n_gold = 0
    for g, rec in zip(gold, preds):
        tokens = g["tokens"]
        gold_set = set((normalize(" ".join(tokens[a:b + 1])), normalize(" ".join(tokens[c:d + 1])), s)
                       for a, b, c, d, s in g["triplets"])
        pred_set = set((normalize(t["aspect_text"]), normalize(t["opinion_text"]), t["sentiment"])
                       for t in rec["pred"])
        n_correct += len(gold_set & pred_set)
        n_pred += len(pred_set)
        n_gold += len(gold_set)
    return n_correct, n_pred, n_gold


def count_index(gold, preds):
    """So trên chỉ số. Trả về None nếu file không có chỉ số (mô hình sinh văn bản)."""
    n_correct = n_pred = n_gold = 0
    for g, rec in zip(gold, preds):
        if any(t["aspect"] is None for t in rec["pred"]):
            return None
        gold_set = set(g["triplets"])
        pred_set = set((t["aspect"][0], t["aspect"][1], t["opinion"][0], t["opinion"][1], t["sentiment"])
                       for t in rec["pred"])
        n_correct += len(gold_set & pred_set)
        n_pred += len(pred_set)
        n_gold += len(gold_set)
    return n_correct, n_pred, n_gold


def evaluate_file(path, tap=None):
    """Chấm một file dự đoán. Trả về dict các số theo phần trăm; phần chỉ số là None nếu không có."""
    tap, preds = read_preds(path, tap)
    gold = read_gold(tap, "test")
    p, r, f1 = prf(*count_text(gold, preds))
    out = {"tap": tap, "so_cau": len(gold), "test_p": p, "test_r": r, "test_f1": f1,
           "test_p_chi_so": None, "test_r_chi_so": None, "test_f1_chi_so": None}
    counts = count_index(gold, preds)
    if counts is not None:
        out["test_p_chi_so"], out["test_r_chi_so"], out["test_f1_chi_so"] = prf(*counts)
    return out


# ---------------------------------------------------------------- dòng lệnh

def fmt(x):
    return "" if x is None else "{:.2f}".format(x)


def mean_and_half_range(values):
    """Trung bình và một nửa khoảng cách giữa giá trị lớn nhất và nhỏ nhất."""
    return sum(values) / len(values), (max(values) - min(values)) / 2


def evaluate_all(preds_dir):
    """Chấm mọi file {mô hình}/{tập}/seed{n}.jsonl. Trả về danh sách dòng, mỗi dòng một file.

    Bỏ qua thư mục không phải tập ASTE (vd `mvp-dmaste/book`, `mvp-dmaste/pet`):
    chúng có format file giống nhưng đáp án nằm ở bộ dữ liệu khác, chấm bằng
    `models/multi-view-prompting/eval_dmaste.py`.
    """
    rows = []
    for path in sorted(Path(str(preds_dir)).glob("*/*/seed*.jsonl")):
        model, tap = path.parts[-3], path.parts[-2]
        if tap not in DATASETS:
            continue
        try:
            res = evaluate_file(path, tap)
        except FormatError as e:
            print("LỖI {}: {}".format(path, e), file=sys.stderr)
            continue
        res.update({"mo_hinh": model, "seed": int(path.stem[len("seed"):])})
        rows.append(res)
    return rows


def print_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["mo_hinh"], row["tap"])].append(row)
    models = sorted(set(m for m, _ in groups))
    print("F1 trên test (%), so trên văn bản: trung bình ± một nửa khoảng cách giữa các seed")
    print("{:<12}".format("mô hình") + "".join("{:>22}".format(t) for t in DATASETS))
    for model in models:
        cells = []
        for tap in DATASETS:
            runs = groups.get((model, tap))
            if not runs:
                cells.append("-")
                continue
            mean, half = mean_and_half_range([r["test_f1"] for r in runs])
            cells.append("{:.2f} ± {:.2f} (n={})".format(mean, half, len(runs)))
        print("{:<12}".format(model) + "".join("{:>22}".format(c) for c in cells))


def main():
    parser = argparse.ArgumentParser(description="Chấm file dự đoán ASTE theo quy ước chung của nhóm.")
    parser.add_argument("files", nargs="*", help="file dự đoán .jsonl")
    parser.add_argument("--tap", choices=DATASETS, help="tập dữ liệu (mặc định: suy ra từ id)")
    parser.add_argument("--all", action="store_true", help="chấm mọi file trong results/preds")
    parser.add_argument("--preds-dir", default=str(PREDS_DIR), help="thư mục dùng với --all")
    parser.add_argument("--csv", help="với --all: ghi điểm từng file ra CSV")
    args = parser.parse_args()

    if args.all:
        rows = evaluate_all(args.preds_dir)
        cols = ["mo_hinh", "tap", "seed", "test_p", "test_r", "test_f1",
                "test_p_chi_so", "test_r_chi_so", "test_f1_chi_so"]
        print("\t".join(cols))
        for row in rows:
            print("\t".join(fmt(row[c]) if isinstance(row[c], float) or row[c] is None else str(row[c])
                            for c in cols))
        print()
        print_summary(rows)
        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(cols)
                for row in rows:
                    writer.writerow([fmt(row[c]) if isinstance(row[c], float) or row[c] is None
                                     else row[c] for c in cols])
        return 0

    if not args.files:
        parser.error("cần ít nhất một file, hoặc dùng --all")
    status = 0
    for path in args.files:
        try:
            res = evaluate_file(path, args.tap)
        except FormatError as e:
            print("LỖI {}: {}".format(path, e), file=sys.stderr)
            status = 1
            continue
        print("{}  ({}, {} câu)".format(path, res["tap"], res["so_cau"]))
        print("  văn bản:  P {}  R {}  F1 {}".format(fmt(res["test_p"]), fmt(res["test_r"]), fmt(res["test_f1"])))
        if res["test_f1_chi_so"] is None:
            print("  chỉ số:   không có (file không ghi chỉ số)")
        else:
            print("  chỉ số:   P {}  R {}  F1 {}".format(
                fmt(res["test_p_chi_so"]), fmt(res["test_r_chi_so"]), fmt(res["test_f1_chi_so"])))
    return status


if __name__ == "__main__":
    sys.exit(main())
