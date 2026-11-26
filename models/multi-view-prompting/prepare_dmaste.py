#!/usr/bin/env python
"""Tải DMASTE và chuyển sang format văn bản của MvP.

DMASTE (Xu et al., Findings ACL 2023) lưu mỗi dòng là

    Câu####[[[a_start, a_end], [o_start, o_end], 'POS'], ...]####Sub-Domain

chỉ số từ 0, tính cả hai đầu, và ``[-1]`` nghĩa là khía cạnh ẩn (implicit). MvP cần

    Câu####[([a_start, a_end], [o_start, o_end], 'POS'), ...]

MvP viết khía cạnh ẩn thành chuỗi ``'it'``; ``extract_spans_para`` đổi ``'it'`` thành
``null`` khi chấm, đúng nghĩa của DMASTE.

Hai chế độ, theo `docs/ke-hoach-trien-khai-aste.md` mục 3.2 và 5:

* ``--keep-implicit``  giữ mọi mẫu (dùng khi muốn số in-domain của paper DMASTE).
* ``--drop-implicit``  bỏ những bộ ba có khía cạnh ẩn. Kế hoạch của nhóm ghi
  "khi tải về cần kiểm tra có khía cạnh ẩn hay không; nếu có thì lọc", và MvP
  không được huấn luyện để sinh khía cạnh ẩn, nên đây là mặc định cho phần
  zero-shot: bỏ bộ ba ẩn, giữ câu nếu còn ít nhất một bộ ba.

Chạy:

    python models/multi-view-prompting/prepare_dmaste.py --repo models/multi-view-prompting
    python models/multi-view-prompting/prepare_dmaste.py --repo . --verify

An toàn khi chạy lại: file đã tải thì dùng lại.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys
import urllib.request

DOMAINS = ["electronics", "fashion", "beauty", "home",
           "book", "pet", "toy", "grocery"]
# Bốn domain đầu có train/dev/test; bốn domain sau chỉ có dev/test (domain đích).
TARGET = ["book", "pet", "toy", "grocery"]
SPLITS = ["train", "dev", "test"]
RAW_URL = "https://raw.githubusercontent.com/NJNLP/DMASTE/main/dataset/{d}/{s}.txt"
RAW_URL_FALLBACK = "https://raw.githubusercontent.com/NJUNLP/DMASTE/main/dataset/{d}/{s}.txt"


def download(raw_dir: pathlib.Path, log=print) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for d in DOMAINS:
        for s in SPLITS:
            dst = raw_dir / f"{d}_{s}.txt"
            if dst.exists() and dst.stat().st_size > 0:
                continue
            ok = False
            for url in (RAW_URL.format(d=d, s=s), RAW_URL_FALLBACK.format(d=d, s=s)):
                try:
                    urllib.request.urlretrieve(url, dst)
                    log(f"tải {dst.name} ({dst.stat().st_size} byte)")
                    ok = True
                    break
                except Exception:                      # noqa: BLE001
                    continue
            if not ok:
                log(f"bỏ qua {d}/{s} (không có split này)")


def parse_raw(line: str):
    parts = line.rstrip("\n").split("####")
    if len(parts) < 2:
        raise ValueError(f"dòng hỏng: {line[:120]!r}")
    return parts[0], ast.literal_eval(parts[1])


def convert_split(src: pathlib.Path, dst: pathlib.Path, drop_implicit: bool) -> dict:
    n = n_trip = n_implicit = n_dropped_sent = n_dropped_trip = 0
    out = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sent, triples = parse_raw(line)
        kept = []
        for aspect, opinion, polarity in triples:
            if aspect == [-1]:
                n_implicit += 1
                if drop_implicit:
                    n_dropped_trip += 1
                    continue
            kept.append((aspect, opinion, polarity))
        if drop_implicit and not kept:
            n_dropped_sent += 1
            continue
        n += 1
        n_trip += len(kept)
        out.append(f"{sent}####{kept}")
    dst.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
    return {"sent": n, "trip": n_trip, "implicit": n_implicit,
            "drop_sent": n_dropped_sent, "drop_trip": n_dropped_trip}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="models/multi-view-prompting")
    ap.add_argument("--raw", default=None,
                    help="nơi giữ file DMASTE gốc (mặc định: <repo>/dmaste_raw, đã gitignore)")
    ap.add_argument("--keep-implicit", dest="drop", action="store_false")
    ap.add_argument("--drop-implicit", dest="drop", action="store_true",
                    help="bỏ bộ ba có khía cạnh ẩn (mặc định, theo kế hoạch nhóm)")
    ap.add_argument("--target-only", action="store_true",
                    help="chỉ chuyển 4 domain đích (book/pet/toy/grocery)")
    ap.add_argument("--domain", action="append", default=None)
    ap.add_argument("--no-download", dest="dl", action="store_false")
    ap.add_argument("--verify", action="store_true")
    ap.set_defaults(drop=True, dl=True)
    args = ap.parse_args(argv)

    repo = pathlib.Path(args.repo).resolve()
    data_root = repo / "data" / "aste"
    if not data_root.exists():
        raise SystemExit(f"không phải repo MvP (thiếu data/aste): {repo}")
    raw = pathlib.Path(args.raw) if args.raw else repo / "dmaste_raw"

    if args.dl:
        download(raw)
    if not raw.exists() or not any(raw.glob("*.txt")):
        raise SystemExit(
            f"không có file DMASTE gốc ở {raw}\n"
            "  - bỏ --no-download để tải, hoặc\n"
            "  - trỏ --raw tới thư mục chứa <domain>_<split>.txt")

    domains = args.domain or (TARGET if args.target_only else DOMAINS)
    total = {"sent": 0, "trip": 0, "implicit": 0, "drop_sent": 0, "drop_trip": 0}
    print("%-12s %-6s %6s %7s %9s %8s %8s" % (
        "domain", "split", "câu", "bộ ba", "ẩn", "bỏ câu", "bỏ bộ ba"))
    found_any = False
    for d in domains:
        out_dir = data_root / f"dmaste_{d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for s in SPLITS:
            src = raw / f"{d}_{s}.txt"
            if not src.exists():
                continue
            found_any = True
            st = convert_split(src, out_dir / f"{s}.txt", args.drop)
            for k in total:
                total[k] += st[k]
            print("%-12s %-6s %6d %7d %9d %8d %8d" % (
                d, s, st["sent"], st["trip"], st["implicit"],
                st["drop_sent"], st["drop_trip"]))
    if not found_any:
        raise SystemExit(f"không tìm thấy {domains} trong {raw} (xem tên file: <domain>_<split>.txt)")
    print("\nTỔNG:", total)
    if args.verify:
        bad = 0
        for d in domains:
            for s in SPLITS:
                p = data_root / f"dmaste_{d}" / f"{s}.txt"
                if not p.exists():
                    continue
                for line in p.read_text(encoding="utf-8").splitlines():
                    if "####" not in line:
                        bad += 1
        print("kiểm tra định dạng:", "OK" if bad == 0 else f"{bad} dòng hỏng")
    return 0


if __name__ == "__main__":
    sys.exit(main())
