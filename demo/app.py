#!/usr/bin/env python
"""Gradio demo cho phần MvP (docs/ke-hoach-trien-khai-aste.md, giai đoạn 5).

Bố cục theo `phan-cong-cong-viec.md` mục 6 (rủi ro): **MvP chạy trực tiếp**, các
mô hình còn lại hiển thị dự đoán đã lưu sẵn trong `results/preds/`. Nhờ vậy demo
không cần nạp bốn bộ thư viện khác nhau trong cùng một tiến trình, mà vẫn so sánh
được bốn mô hình cạnh nhau trên cùng một câu.

Vì ba mô hình kia dùng dự đoán đã lưu, chúng chỉ trả lời được những câu có trong
tập test; câu khác sẽ báo "không có trong tập test". MvP thì câu nào cũng chạy.

Chạy trên Kaggle, trong venv Python 3.8 có torch/transformers:

    python demo/app.py --share
    python demo/app.py --self-test          # kiểm tra backend, không cần Gradio

MvP lấy checkpoint ở `results/runs/mvp/*/seed*/final` (hoặc `--runs`).
Không import `src/main.py`: file đó kéo PyTorch Lightning, mà Lightning 1.9.5
xung đột pydantic của Gradio 4. Phần giải mã có ràng buộc được chép nguyên từ
`src/main.py` nên kết quả trùng với lúc chấm điểm.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from collections import Counter
from functools import partial

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREDS = ROOT / "results" / "preds"
DEFAULT_RUNS = ROOT / "results" / "runs" / "mvp"
POLARITY = {"great": "positive", "bad": "negative", "ok": "neutral",
            "POS": "positive", "NEG": "negative", "NEU": "neutral"}

# id token của t5-base, giống bảng trong src/main.py
TO_ID = {"OT": [667], "AT": [188], "SP": [134], "AC": [254], "SS": [4256],
         "EP": [8569], "[": [784], "]": [908], "it": [34], "null": [206, 195]}

# Nhãn hiển thị của bốn mô hình, theo tên ở quy ước mục 1
MODELS = [("mvp", "MvP (T5-base, chạy trực tiếp)"),
          ("span-aste", "Span-ASTE (BERT)"),
          ("gts-bert", "GTS-BERT"),
          ("emcgcn", "EMC-GCN"),
          ("rule", "Luật cú pháp"),
          ("crf", "CRF + hồi quy logistic")]


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# dự đoán đã lưu của các mô hình khác
# --------------------------------------------------------------------------- #
def load_saved(preds_dir: pathlib.Path = PREDS) -> dict[str, dict[str, list]]:
    """{tên mô hình: {câu (chữ thường, gộp khoảng trắng): [(aspect, opinion, pol)]}}.

    Chỉ đọc `seed0` cho nhẹ; demo không cần cả hai seed.
    """
    saved: dict[str, dict[str, list]] = {}
    for model, _ in MODELS:
        if model == "mvp":
            continue
        found: dict[str, list] = {}
        for path in sorted((preds_dir / model).glob("*/seed0.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = " ".join(rec["tokens"]).lower()
                triplets = []
                for t in rec["pred"]:
                    pol = POLARITY.get(t["sentiment"], t["sentiment"])
                    if t["aspect"] is None:
                        a = t["aspect_text"]
                    else:
                        a = " ".join(rec["tokens"][t["aspect"][0]:t["aspect"][1] + 1])
                    if t["opinion"] is None:
                        o = t["opinion_text"]
                    else:
                        o = " ".join(rec["tokens"][t["opinion"][0]:t["opinion"][1] + 1])
                    triplets.append((a, o, pol))
                found[key] = triplets
        if found:
            saved[model] = found
            _log(f"[demo] {model}: {len(found)} câu có dự đoán lưu sẵn")
    return saved


# --------------------------------------------------------------------------- #
# MvP chạy trực tiếp
# --------------------------------------------------------------------------- #
def _bootstrap(repo: pathlib.Path):
    """Nạp được `t5.py`/`const.py` mà không kéo theo `main.py` (Lightning)."""
    src = repo / "src"
    if not (src / "t5.py").exists():
        raise SystemExit(f"không thấy mã MvP ở {src}")
    sys.path.insert(0, str(src))
    import torch                                              # noqa: PLC0415
    from transformers import T5Tokenizer                      # noqa: PLC0415
    return src, torch, T5Tokenizer


def _ensure_force_tokens(tokenizer, src: pathlib.Path) -> None:
    """const.py mở force_tokens.json ngay khi import; tạo trước nếu thiếu."""
    path = src / "force_tokens.json"
    if path.exists():
        return
    sentiment, special = [], []
    for w in ["great", "ok", "bad"]:
        sentiment.extend(tokenizer(w, return_tensors="pt")["input_ids"].tolist()[0])
    for w in ["[O", "[A", "[S", "[C", "[SS"]:
        special.extend(tokenizer(w, return_tensors="pt")["input_ids"].tolist()[0])
    special = [t for t in special if t != 784]
    path.write_text(json.dumps({"cate_tokens": {}, "all_tokens": {},
                                "sentiment_tokens": {"aste": sentiment},
                                "special_tokens": special}, indent=4),
                    encoding="utf-8")


def prefix_allowed_tokens(force_tokens, source_ids, batch_id, input_ids):
    """Giải mã có ràng buộc — chép từ `src/main.py` (nhánh ASTE)."""
    tip = TO_ID
    left = (input_ids == tip["["][0]).nonzero()
    right = (input_ids == tip["]"][0]).nonzero()
    n_left, n_right = len(left), len(right)
    last_right = right[-1][0] if right.nelement() > 0 else -1
    last_left = left[-1][0] if left.nelement() > 0 else -1
    cur_id = input_ids[-1]

    if cur_id in tip["["]:
        return force_tokens["special_tokens"]
    if cur_id in tip["AT"] + tip["OT"] + tip["EP"] + tip["SP"] + tip["AC"]:
        return tip["]"]
    if cur_id in tip["SS"]:
        return tip["EP"]
    if last_left == -1:
        return tip["["] + [1]
    if (last_left != -1 and last_right == -1) or last_left > last_right:
        return tip["]"]
    cur_term = input_ids[last_left + 1]

    if cur_term in tip["SP"]:
        ret = force_tokens["sentiment_tokens"]["aste"]
    elif cur_term in tip["AT"]:
        ret = source_ids[batch_id].tolist()
    elif cur_term in tip["SS"]:
        return [3] + tip["]"] + [1]
    elif cur_term in tip["AC"]:
        raise ValueError("AC không dùng cho ASTE")
    elif cur_term in tip["OT"]:
        ret = source_ids[batch_id].tolist()
    else:
        raise ValueError(cur_term)

    if n_left == n_right:
        ret = set(ret)
        ret.discard(tip["]"][0])
        for w in force_tokens["special_tokens"]:
            ret.discard(w)
        ret = list(ret)
    elif n_left > n_right:
        ret = ret + tip["]"]
    else:
        raise ValueError
    return ret + tip["["] + [1]


class MvP:
    """Một checkpoint MvP, chạy đúng công thức suy luận của `main.py`."""

    def __init__(self, repo: pathlib.Path, runs: pathlib.Path, device: str | None = None):
        self.src, self.torch, self.T5Tokenizer = _bootstrap(repo)
        self.checkpoints = discover_checkpoints(runs)
        if not self.checkpoints:
            raise SystemExit(f"không có checkpoint nào ở {runs}/*/seed*/final")
        self.device = device or ("cuda:0" if self.torch.cuda.is_available() else "cpu")
        self._cache = {}
        self._force_tokens = None
        _log(f"[demo] MvP: {len(self.checkpoints)} checkpoint, device={self.device}")

    def _load(self, name: str):
        if name in self._cache:
            return self._cache[name]
        from t5 import MyT5ForConditionalGeneration               # noqa: PLC0415
        path = self.checkpoints[name]
        _log(f"[demo] nạp {path} ...")
        tok = self.T5Tokenizer.from_pretrained(path, use_fast=False)
        model = MyT5ForConditionalGeneration.from_pretrained(path)
        model.to(self.device).eval()
        if self._force_tokens is None:
            _ensure_force_tokens(tok, self.src)
            from const import force_tokens                        # noqa: PLC0415
            self._force_tokens = force_tokens
        while len(self._cache) >= 2:      # giữ tối đa 2 checkpoint trong VRAM
            del self._cache[next(iter(self._cache))]
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        self._cache[name] = (model, tok)
        _log("[demo] nạp xong")
        return model, tok

    def predict(self, sentence: str, name: str, k: int = 5,
                constrained: bool = True):
        from const import optim_orders_all                       # noqa: PLC0415
        from eval_utils import extract_spans_para                 # noqa: PLC0415

        model, tok = self._load(name)
        max_len = 200
        tokens = sentence.lower().split()
        orders = optim_orders_all["aste"]["laptop14"][:max(1, min(int(k), 6))]

        raws = []
        with self.torch.no_grad():
            for order in orders:
                prompt = " ".join(tokens + order.split())
                enc = tok.batch_encode_plus([prompt], max_length=max_len,
                                            padding="max_length", truncation=True,
                                            return_tensors="pt")
                src_ids = enc["input_ids"].to(self.device)
                fn = partial(prefix_allowed_tokens, self._force_tokens, src_ids) \
                    if constrained else None
                outs = model.generate(input_ids=src_ids,
                                      attention_mask=enc["attention_mask"].to(self.device),
                                      max_length=max_len, num_beams=1,
                                      early_stopping=True,
                                      return_dict_in_generate=True,
                                      output_scores=True,
                                      prefix_allowed_tokens_fn=fn)
                raws.append(tok.decode(outs.sequences[0], skip_special_tokens=True))

        quads = []
        for r in raws:
            quads.extend(extract_spans_para(seq=r, seq_type="pred"))
        kept = [q for q, c in Counter(quads).items() if c >= len(raws) / 2]

        triplets = []
        for _ac, at, sp, ot in kept:
            if sp == "null" or ot == "null" or at == "null":
                continue
            triplets.append((at, ot, POLARITY.get(sp, sp)))
        return sorted(set(triplets)), raws


def discover_checkpoints(runs: pathlib.Path) -> dict[str, str]:
    out = {}
    if runs.exists():
        for final in sorted(runs.glob("*/seed*/final")):
            out[f"{final.parent.parent.name} ({final.parent.name})"] = str(final)
    return out


# --------------------------------------------------------------------------- #
# tô màu span trong câu
# --------------------------------------------------------------------------- #
COLORS = {"positive": "#bbf7d0", "negative": "#fecaca", "neutral": "#fef08a"}


def highlight(sentence: str, triplets: list) -> str:
    """HTML tô màu các span khía cạnh/ý kiến, giữ nguyên dấu câu."""
    tokens = sentence.split()
    tags = [None] * len(tokens)
    lowered = [t.lower().strip(".,!?;:\"'") for t in tokens]
    for aspect, opinion, pol in triplets:
        for text in (aspect, opinion):
            words = [w.lower().strip(".,!?;:\"'") for w in str(text).split()]
            if not words:
                continue
            for i in range(len(lowered) - len(words) + 1):
                if lowered[i:i + len(words)] == words:
                    for j in range(i, i + len(words)):
                        tags[j] = tags[j] or pol
                    break
    parts = []
    for tok, pol in zip(tokens, tags):
        if pol is None:
            parts.append(tok)
        else:
            parts.append(f'<span style="background:{COLORS.get(pol, "#e2e8f0")};'
                         f'padding:1px 3px;border-radius:4px">{tok}</span>')
    return " ".join(parts)


def format_table(triplets: list) -> list[list[str]]:
    return [[a, o, p] for a, o, p in triplets]


# --------------------------------------------------------------------------- #
# Gradio
# --------------------------------------------------------------------------- #
def build_demo(repo: pathlib.Path, runs: pathlib.Path, use_mvp: bool = True):
    import gradio as gr                                        # noqa: PLC0415

    saved = load_saved()
    mvp = None
    if use_mvp:
        try:
            mvp = MvP(repo, runs)
        except SystemExit as exc:
            _log(f"[demo] MvP không chạy được ({exc}) — chỉ hiện dự đoán lưu sẵn")

    choices = list(mvp.checkpoints) if mvp else []

    def run(sentence, ckpt, k, constrained, trim):
        if not sentence or not sentence.strip():
            return "", "Hãy nhập một câu.", []

        key = " ".join(sentence.lower().split())
        rows, notes, html = [], [], []
        for model, label in MODELS:
            if model == "mvp":
                if mvp is None:
                    continue
                triplets, raws = mvp.predict(sentence, ckpt, k, constrained)
                extra = f"{len(raws)} góc nhìn, bỏ phiếu ≥ {len(raws) / 2:.1f}"
            else:
                triplets = saved.get(model, {}).get(key)
                if triplets is None:
                    notes.append(f"{label}: câu không có trong tập test")
                    continue
                extra = f"dự đoán lưu sẵn, {len(triplets)} bộ ba"
            if trim:
                triplets = triplets[:trim]
            rows.extend([[label, a, o, p] for a, o, p in triplets])
            html.append(f"<p><b>{label}</b> <small>({extra})</small><br>"
                        f"{highlight(sentence, triplets)}</p>")

        msg = "\n".join(notes) if notes else "Đã chạy xong."
        return "\n".join(html), msg, rows

    with gr.Blocks(title="ASTE — so sánh bốn mô hình") as demo:
        gr.Markdown(
            "# Trích xuất bộ ba khía cạnh – ý kiến – cảm xúc\n"
            "MvP chạy trực tiếp; các mô hình còn lại hiện dự đoán đã lưu sẵn trên "
            "tập test (nên chỉ trả lời được câu có trong tập test).")
        with gr.Row():
            with gr.Column(scale=3):
                sentence = gr.Textbox(
                    label="Câu đánh giá", lines=3,
                    value="Battery life is great and the keyboard feels solid.")
                with gr.Row():
                    ckpt = gr.Dropdown(choices, value=choices[0] if choices else None,
                                       label="Checkpoint MvP")
                    k = gr.Slider(1, 6, value=5, step=1, label="Số góc nhìn K")
                with gr.Row():
                    constrained = gr.Checkbox(value=True, label="Giải mã có ràng buộc")
                    trim = gr.Slider(0, 10, value=0, step=1,
                                     label="Giới hạn số bộ ba (0 = không)")
                button = gr.Button("Trích xuất", variant="primary")
            with gr.Column(scale=5):
                rendered = gr.HTML(label="Câu đã tô màu span")
                status = gr.Textbox(label="Trạng thái", interactive=False, lines=4)
                table = gr.Dataframe(headers=["Mô hình", "Khía cạnh", "Ý kiến", "Cảm xúc"],
                                     datatype=["str"] * 4, interactive=False,
                                     label="Bộ ba dự đoán")
        button.click(run, [sentence, ckpt, k, constrained, trim],
                     [rendered, status, table])
        sentence.submit(run, [sentence, ckpt, k, constrained, trim],
                        [rendered, status, table])
    return demo


SELF_TEST = {
    "laptop14": "Battery life is great and the keyboard feels solid.",
    "rest14": "The staff was very friendly and the pasta was delicious, "
              "but the room was too noisy.",
}


def self_test(args) -> int:
    """Chạy một câu qua MvP, không cần Gradio — để kiểm tra backend."""
    t0 = time.time()
    mvp = MvP(pathlib.Path(args.repo), pathlib.Path(args.runs))
    name = args.model or next(iter(mvp.checkpoints))
    sentence = args.sentence or SELF_TEST["rest14"]

    _log(f"[self-test] checkpoint = {name}")
    _log(f"[self-test] sentence   = {sentence}")
    triplets, raws = mvp.predict(sentence, name, args.views, not args.no_constrained)
    for i, r in enumerate(raws, 1):
        _log(f"[self-test] góc nhìn {i}: {r}")
    for a, o, p in triplets:
        _log(f"  {a!r:30} {o!r:25} {p}")
    _log(f"\n[self-test] {time.time() - t0:.1f}s")
    if not triplets:
        _log("[self-test] FAIL: 0 bộ ba")
        return 1
    _log("[self-test] PASS")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(ROOT / "models" / "multi-view-prompting"))
    ap.add_argument("--runs", default=str(DEFAULT_RUNS))
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-mvp", action="store_true",
                    help="chỉ hiện dự đoán lưu sẵn (không nạp checkpoint)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--sentence", default=None)
    ap.add_argument("--views", type=int, default=5)
    ap.add_argument("--no-constrained", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(args)

    demo = build_demo(pathlib.Path(args.repo), pathlib.Path(args.runs),
                      use_mvp=not args.no_mvp)
    try:
        demo.launch(share=args.share, server_name="0.0.0.0",
                    server_port=args.port, prevent_thread_lock=True)
    except Exception as exc:                                   # noqa: BLE001
        _log(f"[demo] launch lỗi ({exc.__class__.__name__}: {exc})")
        if args.share:
            _log("[demo] thử lại không có --share")
            demo.launch(share=False, server_name="0.0.0.0",
                        server_port=args.port, prevent_thread_lock=True)
        else:
            raise
    _log(f"LOCAL URL: {demo.local_url}")
    share_url = getattr(demo, "share_url", None)
    _log(f"PUBLIC URL: {share_url}" if share_url else
         "PUBLIC URL: không tạo được tunnel (kiểm tra Internet / --share)")
    demo.block_thread()
    return 0


if __name__ == "__main__":
    sys.exit(main())
