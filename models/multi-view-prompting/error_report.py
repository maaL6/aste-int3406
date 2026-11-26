#!/usr/bin/env python
"""Phân tích lỗi MvP, nhãn loại lỗi theo docs/quy-uoc-chung.md mục 10.

Chạy trên file dự đoán đúng format chung (results/preds/mvp/{tập}/seed{n}.jsonl),
nên số ở đây khớp với `eval/aste_eval.py` và với runs.csv.

Sáu loại lỗi của nhóm:

    Sai biên        đúng vị trí nhưng span thừa/thiếu từ
    Sai cặp         khía cạnh và ý kiến đều đúng nhưng ghép sai với nhau
    Sai cảm xúc     đúng cặp, sai nhãn cảm xúc
    Bỏ sót          bộ ba có trong đáp án nhưng không được dự đoán
    Dự đoán thừa    bộ ba được dự đoán nhưng không có trong đáp án
    Sai định dạng   đầu ra không đọc được thành bộ ba (riêng MvP)

Với MvP, `Sai định dạng` đếm các bộ ba sinh ra mà `extract_spans_para` không tách
được, hoặc có phần tử rỗng. Những bộ ba đó bị bỏ trước khi ghi file dự đoán, nên
chúng không xuất hiện trong `results/preds`. Script vì vậy nhận thêm `--dump`
(đường dẫn dump gốc của MvP) để đếm chúng.

    python models/multi-view-prompting/error_report.py
    python models/multi-view-prompting/error_report.py --dump runs/aste --out MvP_ERRORS.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict

POL = {"pos": "POS", "neg": "NEG", "neu": "NEU",
       "positive": "POS", "negative": "NEG", "neutral": "NEU",
       "POS": "POS", "NEG": "NEG", "NEU": "NEU"}

CATS = ["Sai biên", "Sai cặp", "Sai cảm xúc", "Bỏ sót"]
TAPS = ["14lap", "14res", "15res", "16res"]


def norm(text) -> str:
    return " ".join(str(text).lower().split())


def load_preds(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def triplets_of(rec: dict) -> list[tuple]:
    out = []
    for t in rec["pred"]:
        out.append((norm(t["aspect_text"]), norm(t["opinion_text"]),
                    POL.get(t["sentiment"], t["sentiment"])))
    return out


def overlaps(g: tuple, p: tuple) -> set[str]:
    """Trường nào trùng (giữ lại cho tham khảo; `categorise` không dùng)."""
    hit = set()
    ga, gp, gs = g[0].split(), g[1].split(), g[2]
    pa, pp, ps = p[0].split(), p[1].split(), p[2]
    if set(ga) & set(pa):
        hit.add("aspect")
    if set(gp) & set(pp):
        hit.add("opinion")
    if gs == ps:
        hit.add("polarity")
    return hit


def categorise(g: tuple, p: tuple | None, pred_pairs: set[tuple]) -> str:
    """Gán một bộ ba vàng bị bỏ lỡ vào một trong bốn loại của nhóm.

    `pred_pairs` là tập (aspect, opinion) của mọi dự đoán trong câu, dùng để nhận
    ra "ghép sai cặp": cả hai span đều có trong câu nhưng nối với phần tử khác.
    """
    if p is None:
        return "Bỏ sót"
    ga, gp, gs = g
    pa, pp, ps = p

    # đúng cặp, sai nhãn
    if ga == pa and gp == pp:
        return "Sai cảm xúc"

    aspect_ok = ga == pa
    opinion_ok = gp == pp
    # cùng vị trí nhưng span thừa/thiếu từ (chồng lấn từ)
    aspect_soft = set(ga.split()) & set(pa.split())
    opinion_soft = set(gp.split()) & set(pp.split())

    if aspect_ok and opinion_ok:
        return "Sai biên"
    if aspect_soft and opinion_soft:
        return "Sai biên"                       # cả hai phía lệch biên
    if aspect_ok:
        # khía cạnh đúng, ý kiến khác: ghép sai cặp nếu ý kiến vàng đi với
        # một khía cạnh khác trong dự đoán
        if any(o == gp and a != ga for a, o in pred_pairs):
            return "Sai cặp"
        return "Sai biên"
    if opinion_ok:
        if any(a == ga and o != gp for a, o in pred_pairs):
            return "Sai cặp"
        return "Sai biên"
    if aspect_soft or opinion_soft:
        return "Sai biên"
    return "Bỏ sót"


def analyse(records: list[dict]) -> dict:
    st = {"n_gold": 0, "n_pred": 0, "correct": 0,
          "cats": Counter(), "spurious": 0, "rows": []}
    for rec in records:
        gold = [tuple(t) for t in rec["_gold"]]
        pred = triplets_of(rec)
        st["n_gold"] += len(gold)
        st["n_pred"] += len(pred)
        gc, pc = Counter(gold), Counter(pred)
        st["correct"] += sum((gc & pc).values())
        g_left = list((gc - pc).elements())
        p_left = list((pc - gc).elements())
        pred_pairs = {(a, o) for a, o, _ in pred}

        # ghép tham lam: mỗi bộ ba vàng bỏ lỡ lấy bộ ba dự đoán thừa giống nhất
        used = [False] * len(p_left)
        for g in g_left:
            best_i, best_score = None, -1
            for i, p in enumerate(p_left):
                if used[i]:
                    continue
                score = (g[0] == p[0]) + (g[1] == p[1]) + (g[2] == p[2])
                if score > best_score:
                    best_i, best_score = i, score
            partner = p_left[best_i] if best_i is not None else None
            cat = categorise(g, partner, pred_pairs)
            if partner is not None and cat != "Bỏ sót":
                used[best_i] = True
            st["cats"][cat] += 1
        st["spurious"] += sum(1 for u in used if not u)
        st["rows"].append((rec, gold, pred, g_left, p_left))
    return st


def coverage_by(st: dict, key_fn) -> dict[str, dict]:
    groups: dict[str, dict] = defaultdict(lambda: {"n_gold": 0, "correct": 0,
                                                   "n_pred": 0, "spurious": 0, "n": 0})
    for rec, gold, pred, _gl, _pl in st["rows"]:
        k = key_fn(rec)
        g = groups[k]
        g["n"] += 1
        g["n_gold"] += len(gold)
        g["n_pred"] += len(pred)
        g["correct"] += len(Counter(gold) & Counter(pred))
        g["spurious"] += len(Counter(pred) - Counter(gold))
    for g in groups.values():
        p = 100.0 * g["correct"] / g["n_pred"] if g["n_pred"] else 0.0
        r = 100.0 * g["correct"] / g["n_gold"] if g["n_gold"] else 0.0
        g["p"], g["r"] = p, r
        g["f1"] = 2 * p * r / (p + r) if p + r else 0.0
    return dict(sorted(groups.items()))


def render(tap: str, seed: int, st: dict, n_format: int | None) -> str:
    L = []
    denom = max(1, st["n_gold"] - st["correct"])
    L.append(f"## {tap} seed{seed}")
    L.append("")
    L.append(f"- Bộ ba vàng: **{st['n_gold']}**, dự đoán: **{st['n_pred']}**, "
             f"đúng: **{st['correct']}**")
    p = 100.0 * st["correct"] / st["n_pred"] if st["n_pred"] else 0.0
    r = 100.0 * st["correct"] / st["n_gold"] if st["n_gold"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    L.append(f"- **P = {p:.2f}  R = {r:.2f}  F1 = {f1:.2f}** (khớp tuyệt đối bộ ba)")
    L.append("")
    L.append("| Loại lỗi | Số lượng | % lỗi phía đáp án |")
    L.append("|---|---:|---:|")
    for cat in CATS:
        n = st["cats"].get(cat, 0)
        L.append(f"| {cat} | {n} | {100 * n / denom:.1f}% |")
    L.append(f"| Dự đoán thừa | {st['spurious']} | — |")
    if n_format is not None:
        L.append(f"| Sai định dạng | {n_format} | — |")
    L.append("")

    L.append("### Theo độ dài câu")
    L.append("")
    L.append("| Độ dài (từ) | Câu | Gold | P | R | F1 |")
    L.append("|---|---:|---:|---:|---:|---:|")

    def bucket(rec):
        n = len(rec["tokens"])
        return "≤10" if n <= 10 else ("11–20" if n <= 20 else ("21–30" if n <= 30 else ">30"))

    for k, g in coverage_by(st, bucket).items():
        L.append(f"| {k} | {g['n']} | {g['n_gold']} | {g['p']:.2f} | {g['r']:.2f} | {g['f1']:.2f} |")
    L.append("")

    L.append("### Theo số bộ ba trong câu")
    L.append("")
    L.append("| #bộ ba | Câu | Gold | P | R | F1 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for k, g in coverage_by(st, lambda r: str(len(r["_gold"]))).items():
        L.append(f"| {k} | {g['n']} | {g['n_gold']} | {g['p']:.2f} | {g['r']:.2f} | {g['f1']:.2f} |")
    L.append("")

    L.append("### Ví dụ")
    L.append("")
    shown = 0
    for rec, gold, pred, g_left, p_left in st["rows"]:
        if not g_left or shown >= 4:
            continue
        shown += 1
        L.append(f"**{rec['id']}** — {rec.get('_sent', '')}")
        L.append("")
        L.append(f"- đáp án: `{gold}`")
        L.append(f"- dự đoán: `{pred}`")
        L.append(f"- bỏ sót: `{g_left}` | thừa: `{p_left}`")
        L.append("")
    return "\n".join(L)


def count_format_errors(dump: pathlib.Path, tap: str, seed: int, data: str) -> int | None:
    """Đếm bộ ba MvP sinh ra mà không đọc được thành bộ ba.

    `extract_spans_para` trả về bốn phần tử rỗng khi chuỗi sinh ra thiếu thẻ hoàn
    toàn; ngoài ra một phần tử rỗng (span rỗng) cũng là sai định dạng.  Các bộ ba
    này bị bỏ khi ghi file nên phải đếm từ dump gốc.
    """
    path = dump / data / ("seed%d" % seed) / "predictions" / (
        "aste_%s_test_cd_path5.jsonl" % data)
    if not path.exists():
        return None
    bad = 0
    for rec in load_preds(path):
        for q in rec["pred"]:
            q = [str(x).strip() for x in q]
            if len(q) != 4 or all(x == "" for x in q):
                bad += 1            # không tách được
            elif q[1] == "" or q[3] == "":
                bad += 1            # span rỗng
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="MvP_ERRORS.md")
    ap.add_argument("--dump", default=None,
                    help="thư mục runs/aste của MvP, để đếm Sai định dạng")
    ap.add_argument("--dataset", default=None, help="chỉ một tập")
    args = ap.parse_args(argv)

    root = pathlib.Path(args.root).resolve() if args.root else \
        pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "eval"))
    from aste_eval import read_gold  # noqa: E402

    RENAME = {"14lap": "laptop14", "14res": "rest14", "15res": "rest15", "16res": "rest16"}
    taps = [args.dataset] if args.dataset else TAPS
    dump = pathlib.Path(args.dump) if args.dump else None

    out = [f"# Phân tích lỗi MvP (C)\n",
           "Sinh bởi `models/multi-view-prompting/error_report.py` từ file dự đoán "
           "trong `results/preds/mvp/`. Nhãn loại lỗi theo "
           "`docs/quy-uoc-chung.md` mục 10. Số P/R/F1 ở đây khớp với "
           "`eval/aste_eval.py` vì cùng đọc một file.\n"]
    for tap in taps:
        gold = read_gold(tap, "test")
        for seed in (0, 1):
            path = root / "results" / "preds" / "mvp" / tap / ("seed%d.jsonl" % seed)
            if not path.exists():
                continue
            recs = load_preds(path)
            for rec, g in zip(recs, gold):
                rec["_gold"] = [
                    (" ".join(g["tokens"][a:b + 1]).lower(),
                     " ".join(g["tokens"][c:d + 1]).lower(), s)
                    for a, b, c, d, s in g["triplets"]
                ]
                rec["_sent"] = " ".join(g["tokens"])
            st = analyse(recs)
            n_fmt = count_format_errors(dump, tap, seed, RENAME[tap]) if dump else None
            out.append(render(tap, seed, st, n_fmt))
            print("%-6s seed%d F1 %.2f | %s" % (
                tap, seed,
                0.0 if not st["n_pred"] else 100.0 * st["correct"] / st["n_pred"],
                dict(st["cats"])))

    dst = pathlib.Path(args.out)
    dst.write_text("\n".join(out), encoding="utf-8")
    print("\nđã ghi", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
