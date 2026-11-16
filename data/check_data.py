"""Kiểm tra dữ liệu đi kèm trong từng repo mô hình trùng với ASTE-Data-V2 gốc.

So sánh theo nội dung: mỗi câu được đưa về (câu, danh sách bộ ba đã sắp xếp), mỗi bộ ba là
(aspect_start, aspect_end, opinion_start, opinion_end, sentiment) với chỉ số từ 0, bao gồm hai đầu.
Hai tập được coi là trùng khi multiset các câu như vậy bằng nhau (không phụ thuộc thứ tự câu).

Chạy từ thư mục gốc của repo:  python data/check_data.py
"""

import ast
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V2_DIR = ROOT / "data" / "ASTE-Data-V2"
MODELS = ROOT / "models"

DATASETS = ["14lap", "14res", "15res", "16res"]
SPLITS = ["train", "dev", "test"]

# Tên thư mục dữ liệu khác nhau giữa các repo
EMCGCN_NAMES = {"14lap": "lap14", "14res": "res14", "15res": "res15", "16res": "res16"}
MVP_NAMES = {"14lap": "laptop14", "14res": "rest14", "15res": "rest15", "16res": "rest16"}

SENTIMENT = {"positive": "POS", "negative": "NEG", "neutral": "NEU"}


def span(indices):
    return min(indices), max(indices)


def parse_line(sentence, triplets_str):
    triplets = []
    for aspect, opinion, sentiment in ast.literal_eval(triplets_str):
        triplets.append((*span(aspect), *span(opinion), sentiment))
    return sentence.strip(), tuple(sorted(triplets))


def read_v2_lines(lines):
    """Định dạng gốc `sentence####[...]`, dùng cho V2 và MvP."""
    out = []
    for line in lines:
        if line.strip():
            sentence, triplets_str = line.rsplit("####", 1)
            out.append(parse_line(sentence, triplets_str))
    return out


def read_span_aste_lines(lines):
    """Span-ASTE thêm hai trường rỗng: `sentence#### #### ####[...]`."""
    out = []
    for line in lines:
        if line.strip():
            parts = line.split("####")
            out.append(parse_line(parts[0], parts[-1]))
    return out


def tags_to_spans(tag_str):
    """`word\\B word\\I word\\O ...` -> danh sách (start, end)."""
    spans, start = [], None
    tags = [tok.rsplit("\\", 1)[1] for tok in tag_str.split(" ")]
    for i, tag in enumerate(tags + ["O"]):
        if start is not None and tag != "I":
            spans.append((start, i - 1))
            start = None
        if tag == "B":
            start = i
    return spans


def read_bio_json(path):
    """Định dạng JSON với nhãn BIO, dùng cho EMC-GCN và GTS."""
    out = []
    for item in json.loads(path.read_text()):
        triplets = []
        for triple in item["triples"]:
            aspects = tags_to_spans(triple["target_tags"])
            opinions = tags_to_spans(triple["opinion_tags"])
            assert len(aspects) == 1 and len(opinions) == 1, (path, item["sentence"])
            triplets.append((*aspects[0], *opinions[0], SENTIMENT[triple["sentiment"]]))
        out.append((item["sentence"].strip(), tuple(sorted(triplets))))
    return out


def load_sources(dataset, split):
    """Trả về {tên nguồn: danh sách câu} cho một tập và một phần chia."""
    sources = {
        "EMC-GCN (D2 .txt)": lambda: read_v2_lines(
            (MODELS / "EMCGCN-ASTE/data/D2" / EMCGCN_NAMES[dataset] / f"{split}.txt").read_text().splitlines()),
        "EMC-GCN (D2 .json)": lambda: read_bio_json(
            MODELS / "EMCGCN-ASTE/data/D2" / EMCGCN_NAMES[dataset] / f"{split}.json"),
        "GTS (ASTE_DATA_V2)": lambda: read_bio_json(
            MODELS / "GTS/data/ASTE_DATA_V2" / EMCGCN_NAMES[dataset] / f"{split}.json"),
        "MvP": lambda: read_v2_lines(
            (MODELS / "multi-view-prompting/data/aste" / MVP_NAMES[dataset] / f"{split}.txt").read_text().splitlines()),
        "Span-ASTE (data.zip)": lambda: read_span_aste_lines(
            zipfile.ZipFile(MODELS / "Span-ASTE/data.zip")
            .read(f"data/triplet_data/{dataset}/{split}.txt").decode().splitlines()),
    }
    return {name: load() for name, load in sources.items()}


def main():
    all_ok = True
    print(f"{'Tập':<12}{'Câu':>6}{'Bộ ba':>7}  Kết quả")
    for dataset in DATASETS:
        for split in SPLITS:
            gold = read_v2_lines((V2_DIR / dataset / f"{split}_triplets.txt").read_text().splitlines())
            gold_counter = Counter(gold)
            n_triplets = sum(len(t) for _, t in gold)
            mismatches = []
            for name, data in load_sources(dataset, split).items():
                if Counter(data) != gold_counter:
                    diff = len(gold_counter - Counter(data)) + len(Counter(data) - gold_counter)
                    mismatches.append(f"{name}: lệch {diff} câu")
            all_ok &= not mismatches
            status = "OK (5/5 nguồn trùng)" if not mismatches else "; ".join(mismatches)
            print(f"{dataset + '/' + split:<12}{len(gold):>6}{n_triplets:>7}  {status}")
    print("\nTất cả nguồn trùng với ASTE-Data-V2." if all_ok else "\nCÓ NGUỒN KHÔNG TRÙNG, xem chi tiết ở trên.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
