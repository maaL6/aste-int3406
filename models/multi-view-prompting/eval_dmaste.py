#!/usr/bin/env python
"""Chấm MvP trên DMASTE, cùng cách so với `eval/aste_eval.py` của nhóm.

Nhóm chưa có script chấm DMASTE (`eval/aste_eval.py` chỉ nhận 14lap/14res/15res/16res),
nên file này giữ đúng tinh thần của script đó: so trên văn bản, chữ thường, gộp
khoảng trắng, một bộ ba đúng khi khớp cả khía cạnh, ý kiến và cảm xúc; P/R/F1 tính
gộp trên cả tập (micro).

Đầu vào là file dự đoán do `export_preds.py --dmaste` ghi
(`results/preds/mvp-dmaste/{domain}/seed{n}.jsonl`) và đáp án ở
`models/multi-view-prompting/data/aste/dmaste_{domain}/test.txt` (đã qua
`prepare_dmaste.py`, mặc định bỏ bộ ba có khía cạnh ẩn).

    python models/multi-view-prompting/eval_dmaste.py
    python models/multi-view-prompting/eval_dmaste.py --domain book
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

# 4 domain đích (chỉ có dev/test) — đây là phần nhóm chạy zero-shot.
# 4 domain in-domain của paper (electronics/fashion/beauty/home) cần train riêng,
# không nằm trong phạm vi nên để ngoài danh sách mặc định.
DOMAINS = ["book", "pet", "toy", "grocery"]
IN_DOMAIN = ["electronics", "fashion", "beauty", "home"]
POL = {"great": "POS", "bad": "NEG", "ok": "NEU",
       "pos": "POS", "neg": "NEG", "neu": "NEU",
       "POS": "POS", "NEG": "NEG", "NEU": "NEU"}


def norm(text) -> str:
    return " ".join(str(text).lower().split())


def read_gold(path: pathlib.Path) -> list[set]:
    """Mỗi dòng -> tập bộ ba (aspect_text, opinion_text, POS|NEG|NEU)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sent, triples = line.rsplit("####", 1)
        tokens = sent.split()
        items = set()
        for aspect, opinion, polarity in ast.literal_eval(triples):
            if aspect == [-1]:
                continue                       # bộ ba ẩn: không chấm ở chế độ này
            at = " ".join(tokens[min(aspect):max(aspect) + 1])
            ot = " ".join(tokens[min(opinion):max(opinion) + 1])
            items.add((norm(at), norm(ot), POL.get(polarity, polarity)))
        out.append(items)
    return out


def read_preds(path: pathlib.Path) -> list[set]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out.append({(norm(t["aspect_text"]), norm(t["opinion_text"]),
                     POL.get(t["sentiment"], t["sentiment"])) for t in rec["pred"]})
    return out


def prf(tp: int, npred: int, ngold: int):
    p = 100.0 * tp / npred if npred else 0.0
    r = 100.0 * tp / ngold if ngold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(gold: list[set], pred: list[set]) -> dict:
    if len(gold) != len(pred):
        raise SystemExit(f"số câu lệch: gold {len(gold)} vs pred {len(pred)}")
    tp = npred = ngold = 0
    for g, p in zip(gold, pred):
        tp += len(g & p)
        npred += len(p)
        ngold += len(g)
    p_, r_, f_ = prf(tp, npred, ngold)
    return {"p": p_, "r": r_, "f1": f_, "tp": tp, "n_pred": npred, "n_gold": ngold}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--domain", action="append", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None, help="ghi bảng ra CSV")
    args = ap.parse_args(argv)

    root = pathlib.Path(args.root).resolve() if args.root else \
        pathlib.Path(__file__).resolve().parents[2]
    data = root / "models" / "multi-view-prompting" / "data" / "aste"
    preds_root = root / "results" / "preds" / "mvp-dmaste"

    rows = []
    print("%-12s %6s %6s %7s %7s %7s %6s" % (
        "domain", "câu", "gold", "P", "R", "F1", "dự đoán"))
    for dom in args.domain or DOMAINS:
        gold_p = data / f"dmaste_{dom}" / "test.txt"
        pred_p = preds_root / dom / ("seed%d.jsonl" % args.seed)
        if not gold_p.exists():
            print("%-12s  thiếu data: %s" % (dom, gold_p))
            continue
        if not pred_p.exists():
            print("%-12s  thiếu dự đoán: %s" % (dom, pred_p))
            continue
        gold, pred = read_gold(gold_p), read_preds(pred_p)
        s = score(gold, pred)
        rows.append((dom, len(gold), s))
        print("%-12s %6d %6d %7.2f %7.2f %7.2f %6d" % (
            dom, len(gold), s["n_gold"], s["p"], s["r"], s["f1"], s["n_pred"]))

    if rows:
        tp = sum(s["tp"] for _, _, s in rows)
        npred = sum(s["n_pred"] for _, _, s in rows)
        ngold = sum(s["n_gold"] for _, _, s in rows)
        p_, r_, f_ = prf(tp, npred, ngold)
        print("%-12s %6s %6d %7.2f %7.2f %7.2f %6d" % (
            "TỔNG", "", ngold, p_, r_, f_, npred))

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("domain,so_cau,gold,p,r,f1,pred\n")
            for dom, n, s in rows:
                f.write("%s,%d,%d,%.2f,%.2f,%.2f,%d\n" % (
                    dom, n, s["n_gold"], s["p"], s["r"], s["f1"], s["n_pred"]))
        print("\nđã ghi", args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
