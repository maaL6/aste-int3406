#!/usr/bin/env python
"""Convert MvP per-sentence dumps into the group prediction format.

MvP writes one JSONL record per test sentence with the generated string and the
parsed quads::

    {"sent": ..., "gold_raw": ..., "pred_raw": ...,
     "gold": [[ac, at, sp, ot], ...], "pred": [[ac, at, sp, ot], ...]}

``docs/quy-uoc-chung.md`` section 4 asks for::

    {"id": "14lap-test-0", "tokens": [...],
     "pred": [{"aspect": null, "opinion": null, "sentiment": "POS",
               "aspect_text": "battery", "opinion_text": "long"}]}

MvP is a generative model, so it has no token indices: ``aspect``/``opinion`` are
``null`` and only the text fields are filled. The sentiment labels MvP emits
(``great``/``bad``/``ok``) are mapped to ``POS``/``NEG``/``NEU``.

Usage
-----
    # 4 ASTE-Data-V2 sets, 2 seeds
    python export_mvp_preds.py --runs runs/aste --out ../results

    # DMASTE zero-shot, checkpoint of laptop14 seed0 on the 4 target domains
    python export_mvp_preds.py --dmaste runs/aste/laptop14/seed0 --domain book

Both write ``<out>/preds/mvp/<tap>/seed<n>.jsonl`` and score each file with
``eval/aste_eval.py`` when it is importable, so the numbers in the report come
from the shared script (section 5 of the conventions).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

# MvP polarity -> group polarity (conventions section 1)
POLARITY = {"great": "POS", "bad": "NEG", "ok": "NEU",
            "POS": "POS", "NEG": "NEG", "NEU": "NEU"}
# repo dataset name -> group dataset name
RENAME = {"laptop14": "14lap", "rest14": "14res", "rest15": "15res", "rest16": "16res"}
ASTE = ["laptop14", "rest14", "rest15", "rest16"]
DMASTE_TARGET = ["book", "pet", "toy", "grocery"]

APOS = re.compile(r"\s+'")


def norm(text: str) -> str:
    """Same as ``aste_eval.normalize`` plus apostrophe folding.

    MvP decodes to detokenised text (``wasn't``, ``jukebox's``) while the gold
    sentence is token-joined (``was n't``, ``juke box 's``), so without folding
    every such triplet is a miss even when it is correct.  Measured effect on
    the shipped dumps: +0.10 F1 on 14res, +0.57 on 16res, 0 elsewhere.
    """
    t = " ".join(text.lower().split())
    t = APOS.sub("'", t)
    return t.replace("' s ", "'s ").replace(" 's", "'s")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def align(tokens: list[str], text: str):
    """Return ``(start, end)`` of ``text`` inside ``tokens``, or ``None``.

    MvP generates detokenised text: ``wasn't`` against the token list
    ``["was", "n't", ...]``, ``jukebox's`` against ``["jukebox", "'s", ...]``.
    Searching with the apostrophe folded on both sides finds the span, and the
    file then carries the token-joined form, so both sides of the comparison
    come from the same token list (conventions section 4).
    """
    target = norm(text)
    if not target:
        return None
    n = len(tokens)
    for a in range(n):
        for b in range(a, n):
            if norm(" ".join(tokens[a:b + 1])) == target:
                return a, b
    return None


def to_triplets(rec: dict, tokens: list[str]) -> tuple[list[tuple[str, str, str]], int]:
    """Dump record -> ([(aspect_text, opinion_text, POS|NEG|NEU)], n_unaligned).

    Quads with ``null`` (missing element, or an implicit aspect that survived as
    ``it`` in a non-DMASTE run) are dropped: the group format has no way to
    express them and ASTE-Data-V2 contains none.  Spans that cannot be found in
    the sentence (a word the model invented) are kept verbatim and counted.
    """
    out, seen, unaligned = [], set(), 0
    for ac, at, sp, ot in rec["pred"]:
        sp = POLARITY.get(sp)
        if sp is None or at in (None, "null", "") or ot in (None, "null", ""):
            continue
        for name, txt in (("aspect", at), ("opinion", ot)):
            span = align(tokens, txt)
            if span is None:
                unaligned += 1
            elif name == "aspect":
                at = " ".join(tokens[span[0]:span[1] + 1])
            else:
                ot = " ".join(tokens[span[0]:span[1] + 1])
        key = (norm(at), norm(ot), sp)
        if key in seen:
            continue
        seen.add(key)
        out.append((at, ot, sp))
    return out, unaligned


def build_records(dump: pathlib.Path, tokens_by_sent: dict[str, list[str]],
                  ids_by_sent: dict[str, list[str]]) -> tuple[list[dict], int]:
    """Attach the group ``id`` (dump order may differ from file order)."""
    used: dict[str, int] = {}
    unaligned = 0
    records = []
    for rec in read_jsonl(dump):
        sent = " ".join(rec["sent"].split())
        ids = ids_by_sent[sent]
        n = used.get(sent, 0)
        if n >= len(ids):
            raise SystemExit(f"{dump.name}: sentence used more often than in gold: {sent[:80]!r}")
        used[sent] = n + 1
        tokens = tokens_by_sent[sent]
        triplets, bad = to_triplets(rec, tokens)
        unaligned += bad
        records.append({"id": ids[n], "tokens": tokens,
                        "pred": [{"aspect": None, "opinion": None, "sentiment": sp,
                                  "aspect_text": at, "opinion_text": ot}
                                 for at, ot, sp in triplets]})
    return records, unaligned


def index_gold(tap: str, eval_dir: pathlib.Path):
    """{sentence: tokens} and {sentence: [ids]} from the group gold files."""
    sys.path.insert(0, str(eval_dir))
    from aste_eval import read_gold  # noqa: E402
    tokens_by_sent, ids_by_sent = {}, {}
    for item in read_gold(tap, "test"):
        key = " ".join(item["tokens"])
        tokens_by_sent[key] = item["tokens"]
        ids_by_sent.setdefault(key, []).append(item["id"])
    return tokens_by_sent, ids_by_sent


def write(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def score(path: pathlib.Path, tap: str, eval_dir: pathlib.Path) -> dict:
    sys.path.insert(0, str(eval_dir))
    from aste_eval import evaluate_file  # noqa: E402
    return evaluate_file(path, tap)


def export_aste(runs: pathlib.Path, out: pathlib.Path, eval_dir: pathlib.Path,
                k: int = 5) -> list[tuple]:
    """Write prediction files and score them.

    K = 5 goes to ``results/preds/mvp/`` (the official runs). Other K values are
    the view ablation and go to ``results/ablation/mvp-k{k}/`` so that
    ``eval/aste_eval.py --all`` still reports exactly the eight official files.
    """
    if k == 5:
        root = out / "results" / "preds" / "mvp"
    else:
        root = out / "results" / "ablation" / ("mvp-k%d" % k)
    rows = []
    print("%-6s %-5s %7s %7s %7s %6s  %s" % (
        "tap", "seed", "P", "R", "F1", "lech", "file"))
    for data in ASTE:
        tap = RENAME[data]
        tokens_by_sent, ids_by_sent = index_gold(tap, eval_dir)
        for seed in (0, 1):
            # K=5 is written to <run>/predictions; the ablation writes to
            # <run>/k<K>/predictions so that caches of different K do not collide.
            # Accept both layouts: a dump directly under <run>/predictions also
            # happens when every K was written into the same directory (the file
            # name already carries path<K>).
            run_dir = runs / data / ("seed%d" % seed)
            name = "aste_%s_test_cd_path%d.jsonl" % (data, k)
            candidates = [run_dir / "predictions" / name]
            if k != 5:
                candidates.insert(0, run_dir / ("k%d" % k) / "predictions" / name)
            dump = next((c for c in candidates if c.exists()), candidates[0])
            if not dump.exists():
                if k == 5:
                    print("%-6s %-5d %7s %7s %7s %6s  MISSING %s"
                          % (tap, seed, "-", "-", "-", "-", dump))
                continue
            recs, unaligned = build_records(dump, tokens_by_sent, ids_by_sent)
            dst = root / tap / ("seed%d.jsonl" % seed)
            write(dst, recs)
            res = score(dst, tap, eval_dir)
            rows.append((tap, seed, res, unaligned))
            print("%-6s %-5d %7.2f %7.2f %7.2f %6d  %s" % (
                tap, seed, res["test_p"], res["test_r"], res["test_f1"],
                unaligned, dst))
    return rows


def export_dmaste(ckpt_run: pathlib.Path, out: pathlib.Path,
                  eval_dir: pathlib.Path, domains: list[str],
                  seed: int = 0) -> None:
    """Zero-shot: one ASTE-Data-V2 checkpoint applied to a DMASTE target domain."""
    print("%-8s %-5s  %s" % ("domain", "seed", "file"))
    for dom in domains:
        dumps = sorted((ckpt_run / "predictions").glob(
            "aste_dmaste_%s_test*_path5.jsonl" % dom))
        if not dumps:
            print("%-8s %-5d  MISSING dump in %s" % (
                dom, seed, ckpt_run / "predictions"))
            continue
        src = dumps[-1]
        # căn span về token của câu, giống phần ASTE-Data-V2
        records, unaligned, dropped = [], 0, 0
        for rec in read_jsonl(src):
            sent_tokens = rec["sent"].split()
            quads = []
            for ac, at, sp, ot in rec["pred"]:
                sp = POLARITY.get(sp)
                if sp is None:
                    continue
                # MvP-ASTE-TOOLKIT: DMASTE cho phep khia canh an (null), nhung mot
                # bo ba ma CA HAI deu null la bo ba rong: mo hinh khong sinh duoc
                # span nao cho cau qua dai. Giu lai se lam giam precision mot cach
                # gia tao, vi no khong bao gio khop dap an.
                if str(at).strip().lower() == "null" and str(ot).strip().lower() == "null":
                    dropped += 1
                    continue
                for name, txt in (("aspect", at), ("opinion", ot)):
                    span = align(sent_tokens, txt)
                    if span is None:
                        unaligned += 1
                    elif name == "aspect":
                        at = " ".join(sent_tokens[span[0]:span[1] + 1])
                    else:
                        ot = " ".join(sent_tokens[span[0]:span[1] + 1])
                quads.append({"aspect": None, "opinion": None, "sentiment": sp,
                              "aspect_text": at, "opinion_text": ot})
            records.append({"id": "dmaste-%s-test-%d" % (dom, len(records)),
                            "tokens": sent_tokens, "pred": quads})
        dst = out / "results" / "preds" / "mvp-dmaste" / dom / ("seed%d.jsonl" % seed)
        write(dst, records)
        print("%-8s %-5d  %s (%d câu, %d span lệch, %d bộ ba rỗng bỏ)"
              % (dom, seed, dst, len(records), unaligned, dropped))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="runs/aste (or one <data>/seed<n>)")
    ap.add_argument("--out", required=True, help="repo root, results/ is created here")
    ap.add_argument("--eval-dir", default=None, help="default: <out>/eval")
    ap.add_argument("--dmaste", action="store_true",
                    help="treat --runs as one checkpoint dir and export DMASTE dumps")
    ap.add_argument("--domain", action="append", default=None,
                    help="DMASTE domain (repeatable); default: the 4 target domains")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5,
                    help="số góc nhìn của dump (mặc định 5; khác 5 thì ghi vào "
                         "preds/mvp-k{k} để không lẫn với kết quả chính)")
    args = ap.parse_args(argv)

    runs = pathlib.Path(args.runs).resolve()
    out = pathlib.Path(args.out).resolve()
    eval_dir = pathlib.Path(args.eval_dir).resolve() if args.eval_dir else out / "eval"

    if args.dmaste:
        export_dmaste(runs, out, eval_dir, args.domain or DMASTE_TARGET, args.seed)
    else:
        export_aste(runs, out, eval_dir, args.k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
