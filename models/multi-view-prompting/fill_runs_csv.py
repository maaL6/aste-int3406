#!/usr/bin/env python
"""Append MvP rows to results/runs.csv from the files on disk.

Reads, for every (tập, seed):
  * results/preds/mvp/{tập}/seed{n}.jsonl      -> test P/R/F1 via eval/aste_eval.py
  * results/logs/mvp/{tập}/seed{n}.stats.json  -> phut_train, ms_moi_cau, mb_gpu, dev_f1
  * results/logs/mvp/{tập}/seed{n}.log         -> test_f1_repo (dòng "precision: ... F1 = ...")
  * results/runs/mvp/{data}/seed{n}/           -> chỉ để xác nhận run có thật

Rows whose numbers are not on disk yet are written with empty cells and a note in
``ghi_chu``; run this again after copying the Kaggle ``runs/`` folder in and it
fills them.  Never rewrites other models' rows: it removes earlier ``mvp`` rows
and appends fresh ones, so rerunning is safe.

    python models/multi-view-prompting/fill_runs_csv.py
    python models/multi-view-prompting/fill_runs_csv.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import subprocess
import sys

RENAME = {"laptop14": "14lap", "rest14": "14res", "rest15": "15res", "rest16": "16res"}
TAPS = ["14lap", "14res", "15res", "16res"]
SEEDS = [0, 1]
RE_REPO = re.compile(r"precision:\s*([\d.]+)\s*recall:\s*([\d.]+)\s*F1\s*=\s*([\d.]+)")

NOTE = ("fast padding (pad theo batch thay vì 200 token, cùng phép toán, nhanh hơn "
        "~3-4 lần); train 2 GPU song song, mỗi run 1 GPU; dev_f1 là epoch 19 vì MvP "
        "giữ checkpoint epoch cuối, dev chấm bằng giải mã 1 góc nhìn không ràng buộc "
        "nên không so được với test_f1; ms_moi_cau tính cho cả K=5 góc nhìn; mb_gpu "
        "trống vì log không ghi")


def repo_f1(log: pathlib.Path):
    if not log.exists():
        return None
    hits = RE_REPO.findall(log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n"))
    return round(float(hits[-1][2]), 2) if hits else None


def eval_f1(path: pathlib.Path, tap: str, root: pathlib.Path):
    if not path.exists():
        return None, None, None
    code = (
        "import json,sys;sys.path.insert(0,{root!r});"
        "from aste_eval import evaluate_file;"
        "r=evaluate_file({p!r},{t!r});"
        "print(json.dumps([r['test_p'],r['test_r'],r['test_f1']]))"
    ).format(root=str(root / "eval"), p=str(path), t=tap)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if out.returncode != 0:
        return None, None, None
    p, r, f = json.loads(out.stdout.strip().splitlines()[-1])
    return round(p, 2), round(r, 2), round(f, 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="repo root (default: two levels up)")
    ap.add_argument("--nguoi-chay", default="C")
    ap.add_argument("--ngay", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--commit", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = pathlib.Path(args.root).resolve() if args.root else \
        pathlib.Path(__file__).resolve().parents[2]
    csv_path = root / "results" / "runs.csv"

    import datetime
    ngay = args.ngay or datetime.date.today().isoformat()
    commit = args.commit or "chưa commit"

    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    body = [r for r in body if not (r and r[0] == "mvp")]

    new = []
    for tap in TAPS:
        for seed in SEEDS:
            pred = root / "results" / "preds" / "mvp" / tap / ("seed%d.jsonl" % seed)
            log = root / "results" / "logs" / "mvp" / tap / ("seed%d.log" % seed)
            stats_p = root / "results" / "logs" / "mvp" / tap / ("seed%d.stats.json" % seed)
            stats = json.loads(stats_p.read_text(encoding="utf-8")) if stats_p.exists() else {}

            p, r, f1 = eval_f1(pred, tap, root)
            note = NOTE
            missing = [n for n, v in (("preds", p), ("dev_f1", stats.get("dev_f1")),
                                      ("phut_train", stats.get("phut_train")),
                                      ("ms_moi_cau", stats.get("ms_moi_cau")))
                       if v is None]
            if missing:
                note += "; thiếu " + ", ".join(missing) + " (chưa có runs/ của session Kaggle)"

            def cell(v, fmt="{:.2f}"):
                return "" if v is None else fmt.format(v)

            new.append([
                "mvp", tap, str(seed), args.nguoi_chay, ngay, commit,
                cell(stats.get("dev_f1")),                    # dev_f1
                cell(p), cell(r), cell(f1),                   # test P/R/F1 (văn bản)
                "",                                           # test_f1_chi_so: MvP không có chỉ số
                cell(repo_f1(log)),                           # test_f1_repo
                cell(stats.get("phut_train"), "{:.1f}"),      # phut_train
                cell(stats.get("ms_moi_cau")),                # ms_moi_cau
                cell(stats.get("trieu_tham_so")),             # trieu_tham_so
                cell(stats.get("mb_gpu"), "{:.0f}"),          # mb_gpu
                note,
            ])

    if args.dry_run:
        for row in new:
            print(",".join(row))
        return 0

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)
        w.writerows(new)
    print("đã ghi %d dòng mvp vào %s" % (len(new), csv_path))
    for row in new:
        print("  %-6s s%s  dev=%-6s test=%-6s repo=%-6s phut=%-6s ms=%s"
              % (row[1], row[2], row[6] or "-", row[9] or "-", row[11] or "-",
                 row[12] or "-", row[13] or "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
