#!/usr/bin/env python
"""Generate notebooks/mvp.ipynb following the MvP repo's own training instructions.

The upstream repo (ZubinGou/multi-view-prompting) trains with

    bash scripts/run_main.sh

and `docs/phan-cong-cong-viec.md` muc 3 asks us to do exactly one thing:

    "MvP (C): ... Sửa `scripts/run_main.sh` để chạy đủ 4 tập.
     Thêm đoạn ghi file dự đoán (`src/main.py:586`)."

So the notebook calls `models/multi-view-prompting/scripts/run_main.sh`, the patched
upstream script — not a wrapper written by us. Layout of the notebook follows
docs/quy-uoc-chung.md muc 8, same as notebooks/gts-bert.ipynb: declare runs in one
cell, then clone / install / versions / train / write results.

    python models/multi-view-prompting/build_notebook.py
"""

import json
import pathlib

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(True)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(True)})


md("""
# mvp

Huấn luyện MvP (T5-base) trên ASTE-Data-V2 — 4 tập × 2 seed — rồi ablation số góc
nhìn và DMASTE zero-shot. Bật GPU T4 và Internet trong phần cài đặt, rồi chạy bằng
"Save & Run All".

Notebook gọi đúng script huấn luyện của repo gốc, `scripts/run_main.sh` (README của
MvP: *"Train separate models: `bash scripts/run_main.sh`"*). Nhóm chỉ sửa script đó
để chạy đủ 4 tập và ghi file dự đoán — xem `docs/phan-cong-cong-viec.md` mục 3.

**Repo đã public**, nên clone không cần token. Ô 1 vẫn đặt `GIT_TERMINAL_PROMPT=0`
để nếu có sự cố mạng thì báo lỗi ngay thay vì treo chờ nhập username.

MvP cần Python 3.8 với `torch 1.12.1+cu116` và `transformers 4.14.1`; Kaggle mặc định
là Python 3.11 với transformers đời mới nên ô 2 dựng venv riêng bằng `uv`.
""")

code("""
# Khai báo các lần chạy ở đây, không sửa ở ô khác
TAPS = ["14lap", "14res", "15res", "16res"]
SEEDS = [0, 1]

PARALLEL = 2        # số run chạy cùng lúc (1 = tuần tự; 2 cho Kaggle T4 x2)
NUM_PATH = 5        # số góc nhìn K (paper dùng 5)
ABLATION_KS = [1, 2, 3]   # các K cho ablation; K=5 là kết quả train chính
RUN_ABLATION = True
RUN_DMASTE = True
""")

code("""
# 1. Clone repo nhóm (public) và in mã commit
import os, subprocess, pathlib, shutil

BRANCH = "Hezu06/mvp_run"      # đổi thành "main" sau khi branch đã merge
REPO = "maaL6/aste-int3406"
ROOT = "/kaggle/working/aste"
URL = "https://github.com/%s.git" % REPO

# repo public nên clone không cần token; vẫn tắt prompt để nếu có sự cố thì báo lỗi
# ngay thay vì treo chờ nhập username
os.environ["GIT_TERMINAL_PROMPT"] = "0"
GIT_ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0")


def git(*argv, check=True):
    return subprocess.run(["git", "-C", ROOT] + list(argv), check=check,
                          capture_output=True, text=True, env=GIT_ENV)


if pathlib.Path(ROOT, ".git").is_dir() and "aste-int3406" in git(
        "config", "--get", "remote.origin.url", check=False).stdout:
    # chạy lại ô này (hoặc restart kernel mà /kaggle/working còn dữ liệu):
    # fetch + reset, KHÔNG xoá thư mục, để results/ đã train không bị mất
    print("đã có repo -> fetch + về đúng branch", BRANCH)
    git("fetch", "-q", "origin", BRANCH)
    git("checkout", "-q", "-B", BRANCH, "origin/" + BRANCH)
    git("reset", "-q", "--hard", "origin/" + BRANCH)
else:
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT, ignore_errors=True)
    print("$ git clone --branch %s %s" % (BRANCH, URL))
    subprocess.run(["git", "clone", "-q", "--branch", BRANCH, URL, ROOT],
                   check=True, env=GIT_ENV)

%cd /kaggle/working/aste
!git rev-parse --short HEAD
!git log -1 --format="%h %s"

# script huấn luyện của repo gốc phải có mặt
assert pathlib.Path("models/multi-view-prompting/scripts/run_main.sh").exists(), \\
    "thiếu scripts/run_main.sh — clone sai branch?"

# force_tokens.json là file SINH RA lúc chạy, và nội dung phụ thuộc danh sách
# dataset trong const.py. Bản cũ (từ upstream) không có domain DMASTE, nên
# cate_tokens thiếu key -> KeyError khi chạy DMASTE. Xoá để main.py sinh lại.
stale = pathlib.Path("models/multi-view-prompting/src/force_tokens.json")
if stale.exists():
    stale.unlink()
    print("đã xoá force_tokens.json cũ để main.py sinh lại đúng dataset")
""")

code("""
# 2. Cài thư viện, đúng như README của repo gốc (mục Setup)
#    conda create -n mvp python=3.8
#    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1
#    pip install -r requirements.txt
import os, subprocess, sys

ROOT = "/kaggle/working/aste"
VENV = "/kaggle/working/mvpenv"
assert os.path.isdir(ROOT), "chưa clone repo — chạy ô 1 trước"
os.chdir(ROOT)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)

# --seed: uv venv KHÔNG cài pip mặc định, thiếu cờ này thì `python -m pip` báo
# "No module named pip". --seed cài pip + setuptools + wheel vào venv.
fresh = not os.path.exists(VENV + "/bin/python")
if fresh:
    subprocess.run(["uv", "venv", "--seed", "--python", "3.8", VENV], check=True)
PY = VENV + "/bin/python"

if fresh or not os.path.isdir(VENV + "/lib/python3.8/site-packages/torch"):
    subprocess.run(
        [PY, "-m", "pip", "install", "-q",
         "torch==1.12.1+cu116", "torchvision==0.13.1+cu116", "torchaudio==0.12.1",
         "--extra-index-url", "https://download.pytorch.org/whl/cu116"], check=True)
    subprocess.run([PY, "-m", "pip", "install", "-q", "-r",
                    "models/multi-view-prompting/requirements.txt"], check=True)
else:
    print("venv đã có sẵn, bỏ qua cài đặt")
print("venv ready:", PY)
""")

code("""
# 3. Phiên bản và GPU
import shutil, subprocess

ROOT = "/kaggle/working/aste"
PY = "/kaggle/working/mvpenv/bin/python"

# nvidia-smi là công cụ của driver NVIDIA, dùng để xem GPU. Không có nó nghĩa là
# phiên này không được cấp GPU (Kaggle: Settings -> Accelerator -> GPU T4 x2).
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)
else:
    print("!! không tìm thấy nvidia-smi -> phiên này KHÔNG có GPU NVIDIA")

check = (
    "import sys, torch, transformers, pytorch_lightning as pl;"
    "print(sys.version);"
    "print('torch', torch.__version__, '| transformers', transformers.__version__,"
    " '| pl', pl.__version__);"
    "print('cuda available:', torch.cuda.is_available(),"
    " '| so GPU:', torch.cuda.device_count());"
    "sys.exit(0 if torch.cuda.is_available() else 3)"
)
if subprocess.run([PY, "-c", check]).returncode != 0:
    raise AssertionError(
        "Phiên này không có GPU. Train 8 run x 20 epoch trên CPU sẽ mất hàng chục giờ,\\n"
        "nên dừng ở đây. Vào Settings -> Accelerator -> GPU T4 x2 (hoặc P100), Save, "
        "rồi Run All lại.")
print("GPU OK")
""")

md("""
## 4 và 5. Huấn luyện, dự đoán trên test, ghi kết quả

Gọi `scripts/run_main.sh` — script huấn luyện của chính repo MvP, đã sửa để chạy đủ
4 tập và ghi file dự đoán (`docs/phan-cong-cong-viec.md` mục 3). Script tự lặp seed và
tự ghi:

- `results/preds/mvp/{tập}/seed{n}.jsonl` — dự đoán, format chung
- `results/logs/mvp/{tập}/seed{n}.log` và `.stats.json` — log + số đo chi phí
- `results/runs/mvp/{data}/seed{n}/final/` — checkpoint

`PARALLEL = 2` truyền vào script để chạy hai run cùng lúc trên hai GPU (mỗi run một
tiến trình riêng, không DDP), nên kết quả giống hệt chạy tuần tự.
""")

code("""
# 4 và 5. Huấn luyện + dự đoán + xuất file dự đoán cho TAPS × SEEDS.
import os, subprocess

ROOT = "/kaggle/working/aste"
PY = "/kaggle/working/mvpenv/bin/python"
os.chdir(ROOT)

env = dict(os.environ, PYTHON=PY, NUM_PATH=str(NUM_PATH),
           PARALLEL=str(PARALLEL), TOKENIZERS_PARALLELISM="false")

# mỗi tập một tiến trình: run_main.sh nhận "<tập> <seed>", hoặc không tham số thì
# chạy cả 4 tập × 2 seed. Ở đây truyền theo từng tập để in ra kịp thời.
for tap in TAPS:
    for seed in SEEDS:
        print("=" * 20, tap, "seed", seed, flush=True)
        subprocess.run(["bash", "models/multi-view-prompting/scripts/run_main.sh",
                        tap, str(seed)], env=env, check=True)

print("xong", len(TAPS) * len(SEEDS), "run")
""")

md("""
## 5b. Ablation số góc nhìn K

Dùng lại checkpoint seed 0 vừa train, chỉ đổi số góc nhìn lúc suy luận nên chỉ tốn thời
gian sinh. Truyền `MODE=infer NUM_PATH=k SAVE_DIR_TAG=/kK` vào cùng `run_main.sh`, đúng
cách repo gốc tách `eval_unified.sh` (không có `--do_train`) khỏi `run_main.sh`.
Kết quả ghi vào `results/ablation/mvp-k{k}/` để bảng chính (`eval/aste_eval.py --all`)
vẫn chỉ có 8 file chính thức.
""")

code("""
# 5b. Ablation K: suy luận lại trên 4 tập với checkpoint seed 0 (không train).
import os, subprocess

ROOT = "/kaggle/working/aste"
PY = "/kaggle/working/mvpenv/bin/python"
os.chdir(ROOT)

if RUN_ABLATION:
    for k in ABLATION_KS:
        for tap in TAPS:
            ckpt = os.path.join(ROOT, "results/runs/mvp/%s/seed0/final" % {
                "14lap": "laptop14", "14res": "rest14",
                "15res": "rest15", "16res": "rest16"}[tap])
            assert os.path.isdir(ckpt), "chưa có %s — chạy ô 4 trước" % ckpt
            env = dict(os.environ, PYTHON=PY, MODE="infer", CKPT=ckpt,
                       NUM_PATH=str(k), SAVE_DIR_TAG=".k%d" % k,
                       TOKENIZERS_PARALLELISM="false")
            print("--- %s K=%d" % (tap, k), flush=True)
            subprocess.run(["bash", "models/multi-view-prompting/scripts/run_main.sh",
                            tap, "0"], env=env, check=True)
        subprocess.run([PY, "models/multi-view-prompting/export_preds.py",
                        "--runs", "results/runs/mvp", "--out", ".", "--k", str(k)],
                       check=True)
    subprocess.run([PY, "models/multi-view-prompting/ablation_table.py",
                    "--out", "results/MVP_ABLATION.md"], check=True)
else:
    print("bỏ qua ablation (RUN_ABLATION = False)")
""")

md("""
## 5c. DMASTE zero-shot

Kế hoạch nhóm mục 5: dùng checkpoint đã train trên ASTE-Data-V2, chạy thẳng lên bốn
domain đích của DMASTE (`book`, `pet`, `toy`, `grocery`), không train lại.

DMASTE dài hơn ASTE (câu + prompt tới 242 token) nên phải `MAX_SEQ_LENGTH=256`, và
`EVAL_BATCH_SIZE=32` cho khỏi OOM trên T4 15 GB. Bỏ các bộ ba có khía cạnh ẩn theo
dòng "nếu có thì lọc" trong kế hoạch.
""")

code("""
# 5c. DMASTE zero-shot: 1 checkpoint × 4 domain đích, dùng lại run_main.sh.
import os, subprocess

ROOT = "/kaggle/working/aste"
PY = "/kaggle/working/mvpenv/bin/python"
os.chdir(ROOT)

CKPT = os.path.join(ROOT, "results/runs/mvp/laptop14/seed0/final")

if RUN_DMASTE:
    assert os.path.isdir(CKPT), ("chưa có checkpoint laptop14/seed0/final — "
                                 "chạy ô 4 trước")
    subprocess.run([PY, "models/multi-view-prompting/prepare_dmaste.py",
                    "--repo", "models/multi-view-prompting",
                    "--target-only", "--verify"], check=True)

    failed = []
    for dom in ["book", "pet", "toy", "grocery"]:
        print("--- DMASTE", dom, flush=True)
        env = dict(os.environ, PYTHON=PY, MODE="infer", CKPT=CKPT,
                   DATA_OVERRIDE="dmaste_" + dom,
                   MAX_SEQ_LENGTH="256", EVAL_BATCH_SIZE="32",
                   SAVE_DIR_TAG="", TOKENIZERS_PARALLELISM="false")
        rc = subprocess.run(
            ["bash", "models/multi-view-prompting/scripts/run_main.sh",
             "dmaste_" + dom, "0"], env=env).returncode
        if rc != 0:
            failed.append(dom)
            print("!! {} lỗi (exit {}) — xem results/logs/mvp/dmaste_{}/seed0.log"
                  .format(dom, rc, dom))

    assert not failed, "DMASTE lỗi ở: {}".format(", ".join(failed))
    subprocess.run([PY, "models/multi-view-prompting/eval_dmaste.py",
                    "--csv", "results/MVP_DMASTE.csv"], check=True)
else:
    print("bỏ qua DMASTE (RUN_DMASTE = False)")
""")

code("""
# Chấm toàn bộ bằng script đánh giá chung, rồi đóng gói kết quả.
# Zip không gồm results/runs (checkpoint, ~7 GB).
!python eval/aste_eval.py --all
!cd /kaggle/working && zip -qr results.zip aste/results -x 'aste/results/runs/*'

# Checkpoint để tải lên Kaggle Dataset aste-ckpt-mvp:
import pathlib
for p in sorted(pathlib.Path("/kaggle/working/aste/results/runs/mvp").glob("*/seed*/final")):
    print(p)
""")

md("""
## Ghi chú

- `scripts/run_main.sh` là script của repo gốc; nhóm chỉ sửa phần lặp tập/seed và ghi
  kết quả. Tham số huấn luyện giữ nguyên bản gốc: 20 epoch, batch 16, lr 1e-4,
  `--save_top_k 0` (checkpoint epoch cuối), 5 góc nhìn, `vote`, `--constrained_decode`.
- `run_main.sh` ghi `phut_train`, `trieu_tham_so`, `ms_moi_cau`, `mb_gpu` vào
  `results/logs/mvp/{tập}/seed{n}.stats.json` — định nghĩa ở mục 6 của
  `docs/quy-uoc-chung.md`. `mb_gpu` để trống vì `pytorch-lightning` không in
  `max_memory_allocated`.
- File dự đoán do `export_preds.py` ghi: `aspect`/`opinion` là `null`, chỉ có
  `aspect_text`/`opinion_text` (MvP sinh văn bản, không có chỉ số).
- MvP sinh chữ đã nối (`wasn't`), câu gốc tách token (`was n't`), nên `export_preds.py`
  căn span về đúng token trước khi ghi; không có bước này thì mọi bộ ba chứa dấu nháy
  bị tính sai dù dự đoán đúng.
- `ms_moi_cau` tính cho cả K=5 góc nhìn, không phải một lượt sinh.
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = pathlib.Path(__file__).resolve().parents[2] / "notebooks" / "mvp.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("wrote", out, len(CELLS), "cells")
