#!/usr/bin/env python
"""Bảng thí nghiệm số góc nhìn (K) của MvP — giai đoạn 5, việc của C.

Đọc `results/preds/mvp/` (K=5) và `results/ablation/mvp-k{1,2,3}/` (K nhỏ), chấm lại
bằng `eval/aste_eval.py` rồi in bảng markdown. Không đọc số repo in ra.

    python models/multi-view-prompting/ablation_table.py
    python models/multi-view-prompting/ablation_table.py --out results/MVP_ABLATION.md
"""

from __future__ import annotations

import argparse
import pathlib
import sys

TAPS = ["14lap", "14res", "15res", "16res"]
KS = [1, 2, 3, 5]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    root = pathlib.Path(args.root).resolve() if args.root else \
        pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "eval"))
    from aste_eval import evaluate_file  # noqa: E402

    table = {}
    for tap in TAPS:
        table[tap] = {}
        for k in KS:
            base = (root / "results" / "preds" / "mvp" if k == 5
                    else root / "results" / "ablation" / ("mvp-k%d" % k))
            path = base / tap / "seed0.jsonl"
            if not path.exists():
                continue
            r = evaluate_file(path, tap)
            table[tap][k] = (r["test_p"], r["test_r"], r["test_f1"])

    lines = ["| Tập | " + " | ".join("K=%d" % k for k in KS) + " |",
             "|---|" + "---:|" * len(KS)]
    for tap in TAPS:
        cells = []
        for k in KS:
            v = table[tap].get(k)
            cells.append("-" if v is None else "%.2f" % v[2])
        lines.append("| %s | %s |" % (tap, " | ".join(cells)))

    # trung bình cột
    avg = []
    for k in KS:
        vals = [table[t][k][2] for t in TAPS if k in table[t]]
        avg.append("-" if not vals else "%.2f" % (sum(vals) / len(vals)))
    lines.append("| **Trung bình** | %s |" % " | ".join("**%s**" % a for a in avg))
    print("\n".join(lines))

    best = None
    best_avg = -1.0
    for k in KS:
        vals = [table[t][k][2] for t in TAPS if k in table[t]]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        if avg > best_avg:
            best, best_avg = k, avg
    if best is not None:
        print("\nK tốt nhất theo F1 trung bình: K=%d (%.2f, %d tập)"
              % (best, best_avg,
                 sum(1 for t in TAPS if best in table[t])))
    else:
        print("\nkhông có dữ liệu để so sánh K")

    if args.out:
        body = ["# Thí nghiệm số góc nhìn K của MvP\n",
                "Sinh bởi `models/multi-view-prompting/ablation_table.py`; F1 chấm bằng "
                "`eval/aste_eval.py` (so trên văn bản), seed 0, dùng lại checkpoint đã "
                "train nên chỉ tốn thời gian suy luận.\n",
                *lines, ""]
        pathlib.Path(args.out).write_text("\n".join(body), encoding="utf-8")
        print("đã ghi", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
