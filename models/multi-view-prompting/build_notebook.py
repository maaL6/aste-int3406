#!/usr/bin/env python
"""Generate notebooks/mvp.ipynb in the group's 5-step layout.

Run from the repo root:  python models/multi-view-prompting/build_notebook.py

`docs/quy-uoc-chung.md` muc 8 says a notebook may only contain install and run
commands: all model code lives in models/, all orchestration lives in
models/multi-view-prompting/run_all.py. This file produces exactly that, so the
notebook is 7 cells and every one of them is a shell/CLI call.
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

MvP cần Python 3.8 với `torch 1.12` và `transformers 4.14.1`; Kaggle mặc định là
Python 3.11 với transformers đời mới nên ô 2 dựng một venv riêng bằng `uv`.

| Bước | Nội dung | Thời gian (T4 x2) |
|---|---|---|
| 1–3 | clone, cài, kiểm tra môi trường | ~8 phút |
| 4–5 | train 8 run + suy luận + chấm điểm | ~3.5 giờ |
| 5b | ablation K ∈ {1,2,3} | ~45 phút |
| 5c | DMASTE zero-shot, 4 domain đích | ~15 phút |

Toàn bộ nằm trong một hàm điều phối `models/multi-view-prompting/run_all.py`, nên
bị ngắt giữa chừng chỉ cần chạy lại: run nào đã có `final/` thì bỏ qua.

`PARALLEL = 2` chạy hai tiến trình độc lập, mỗi tiến trình một GPU (không phải DDP),
nên kết quả giống hệt chạy tuần tự. Kaggle tính 1 giờ trên 2 GPU là 1 giờ quota, nên
song song chỉ lãi thời gian.
""")

code("""
# Khai báo các lần chạy ở đây, không sửa ở ô khác
TAPS = ["14lap", "14res", "15res", "16res"]
SEEDS = "0,1"
NUM_PATH = 5          # số góc nhìn K khi suy luận (paper dùng 5)
KS = "1,2,3"          # các K cho ablation; K=5 là kết quả train chính
PARALLEL = 2          # số GPU chạy song song (Kaggle T4 x2 -> 2)
RUN_DMASTE = True     # DMASTE zero-shot (kế hoạch nhóm, mục 5)
""")

code("""
# 1. Clone repo nhóm và in mã commit
!git clone -q https://github.com/maaL6/aste-int3406.git /kaggle/working/aste
%cd /kaggle/working/aste
!git rev-parse --short HEAD
""")

code("""
# 2. Cài thư viện: venv Python 3.8 với đúng phiên bản MvP cần
#    (Kaggle mặc định Python 3.11 + transformers đời mới, MvP không chạy được)
import os, subprocess, sys
VENV = "/kaggle/working/mvpenv"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
if not os.path.exists(VENV + "/bin/python"):
    subprocess.run(["uv", "venv", "--python", "3.8", VENV], check=True)
PY = VENV + "/bin/python"
subprocess.run([PY, "-m", "pip", "install", "-q", "pip", "wheel"], check=True)
subprocess.run([PY, "-m", "pip", "install", "-q",
                "torch==1.12.1+cu116", "torchvision==0.13.1+cu116",
                "--extra-index-url", "https://download.pytorch.org/whl/cu116"], check=True)
subprocess.run([PY, "-m", "pip", "install", "-q", "-r",
                "models/multi-view-prompting/requirements.txt"], check=True)
print("venv ready:", PY)
""")

code("""
# 3. Phiên bản và GPU
!nvidia-smi
!/kaggle/working/mvpenv/bin/python -c "import sys, torch, transformers, pytorch_lightning as pl; print(sys.version); print('torch', torch.__version__, '| transformers', transformers.__version__, '| pl', pl.__version__)"
""")

code("""
# 4 và 5. Huấn luyện, dự đoán trên test, xuất file dự đoán, chấm điểm, ablation, DMASTE.
# Mọi quyết định nằm trong models/multi-view-prompting/run_all.py; ô này chỉ truyền
# tham số. Kết quả: results/preds/mvp, results/ablation, results/logs/mvp, runs.csv.
import os, subprocess
env = dict(os.environ, PYTHON="/kaggle/working/mvpenv/bin/python",
           TOKENIZERS_PARALLELISM="false")
cmd = ["/kaggle/working/mvpenv/bin/python",
       "models/multi-view-prompting/run_all.py",
       "--seeds", SEEDS,
       "--ks", KS,
       "--num-path", str(NUM_PATH),
       "--parallel", str(PARALLEL)]
for tap in TAPS:
    cmd += ["--only", tap]
if not RUN_DMASTE:
    cmd += ["--skip-dmaste"]
print("$", " ".join(cmd), flush=True)
subprocess.run(cmd, cwd="/kaggle/working/aste", env=env, check=True)
""")

code("""
# Đóng gói kết quả để tải về (log, dự đoán, bảng). Không gồm checkpoint.
!cd /kaggle/working && zip -qr results.zip aste/results -x 'aste/results/runs/*'
!python eval/aste_eval.py --all

# Checkpoint để tải lên Kaggle Dataset aste-ckpt-mvp, mỗi lần chạy một file:
#   /kaggle/working/aste/results/runs/mvp/<tập>/seed<n>/final
import pathlib
for p in sorted(pathlib.Path("/kaggle/working/aste/results/runs/mvp").glob("*/seed*/final")):
    print(p)
""")

md("""
## Ghi chú

- `run_all.py` ghi `phut_train` (epoch đầu → hết epoch cuối, gồm chấm dev),
  `trieu_tham_so`, `ms_moi_cau` (chỉ phần sinh, không tính nạp mô hình) và `mb_gpu`
  vào `results/logs/mvp/{tập}/seed{n}.stats.json` — đúng định nghĩa mục 6 của
  `docs/quy-uoc-chung.md`.
- File dự đoán do `models/multi-view-prompting/export_preds.py` ghi, đúng format
  mục 4: `aspect`/`opinion` là `null`, chỉ điền `aspect_text`/`opinion_text`.
- MvP sinh chữ đã nối (`wasn't`), câu gốc tách token (`was n't`), nên script xuất file
  căn span về đúng token của câu trước khi ghi. Không có bước này thì mọi bộ ba chứa
  dấu nháy bị tính sai dù dự đoán đúng.
- Ablation ghi vào `results/ablation/mvp-k{k}/` để `eval/aste_eval.py --all` vẫn chỉ
  thấy 8 file chính thức của mô hình.
- `mb_gpu` để trống: `pytorch-lightning` không in `max_memory_allocated`. Muốn có số
  thật thì phải chạy lại một lượt suy luận có gọi hàm đó.
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = pathlib.Path(__file__).resolve().parents[2] / "notebooks" / "mvp.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("wrote", out, len(CELLS), "cells")
