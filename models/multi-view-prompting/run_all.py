#!/usr/bin/env python
"""Điều phối toàn bộ phần MvP: train 4 tập × 2 seed, ablation K, DMASTE zero-shot.

Notebook chỉ gọi script này (`docs/quy-uoc-chung.md`, mục 8: notebook chỉ chứa lệnh
cài và lệnh chạy). Mọi quyết định nằm ở đây và ở `run.sh`.

    python models/multi-view-prompting/run_all.py                 # tất cả
    python models/multi-view-prompting/run_all.py --parallel 2    # 2 GPU song song
    python models/multi-view-prompting/run_all.py --skip-ablation
    python models/multi-view-prompting/run_all.py --only 14lap
    python models/multi-view-prompting/run_all.py --dmaste-checkpoint laptop14/seed0

Mỗi bước đều **idempotent**: run nào đã có `final/` + dump dự đoán thì bỏ qua, nên bị
ngắt giữa chừng chỉ cần chạy lại. In ra thời gian ước tính trước khi làm để biết một
phiên Kaggle có đủ không.

`--parallel N` chạy N tiến trình độc lập, mỗi tiến trình một GPU
(`CUDA_VISIBLE_DEVICES=0..N-1`). Không phải DDP: mỗi run vẫn là một `main.py` riêng,
nên kết quả giống hệt chạy tuần tự. Kaggle tính 1 giờ trên 2 GPU là 1 giờ quota, nên
song song chỉ lãi thời gian.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
DATASETS = {"14lap": "laptop14", "14res": "rest14", "15res": "rest15", "16res": "rest16"}
SEEDS = [0, 1]
# phút huấn luyện đo được trên T4 (results/logs/mvp/*/seed0.stats.json)
EST_TRAIN = {"14lap": 51, "14res": 74, "15res": 33, "16res": 47}
# phút suy luận 5 góc nhìn, đo được: 478-658 ms/câu
EST_TAPS = {"14lap": 328, "14res": 492, "15res": 322, "16res": 326}
DMASTE_DOMAINS = ["book", "pet", "toy", "grocery"]
DMASTE_TEST = {"book": 260, "pet": 253, "toy": 297, "grocery": 299}


def sh(cmd, env=None, check=True):
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    mins = (time.time() - t0) / 60.0
    if check and proc.returncode != 0:
        raise SystemExit("lệnh thất bại (exit %d): %s" % (proc.returncode, " ".join(cmd)))
    print("  -> %.1f phút\n" % mins, flush=True)
    return proc.returncode


def python():
    """Dùng đúng python đang chạy script (venv 3.8 trên Kaggle)."""
    return sys.executable


def count_gpus():
    """Số GPU nhìn thấy; lỗi gì cũng coi như 1 để không chặn chạy."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
        return max(1, len([l for l in out.splitlines() if l.strip()]))
    except Exception:                                          # noqa: BLE001
        return 1


def run_batch(cmd_fn, items, n_gpu, label, extra_env=None):
    """Chạy `cmd_fn(item, gpu)` cho từng item, tối đa `n_gpu` tiến trình cùng lúc.

    Trả về danh sách item thất bại. Mỗi tiến trình một GPU riêng qua
    CUDA_VISIBLE_DEVICES, nên `main.py` (luôn dùng cuda:0) sẽ thấy đúng card của nó.
    """
    extra = extra_env or {}
    if n_gpu <= 1:
        failed = []
        for item in items:
            if sh(cmd_fn(item, 0), env=base_env(0, **extra), check=False) != 0:
                failed.append(item)
        return failed

    queue = list(items)
    running = []          # (proc, item, gpu, t0)
    failed = []
    while queue or running:
        while queue and len(running) < n_gpu:
            item = queue.pop(0)
            # chọn GPU đang rảnh (số thứ tự nhỏ nhất chưa bị chiếm)
            busy = {r[2] for r in running}
            gpu = next(g for g in range(n_gpu) if g not in busy)
            cmd = cmd_fn(item, gpu)
            print("[%s] -> GPU %d: %s" % (label, gpu, " ".join(cmd)), flush=True)
            proc = subprocess.Popen(cmd, cwd=str(ROOT), env=base_env(gpu, **extra),
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            running.append((proc, item, gpu, time.time()))

        # poll ngắn để job ngắn (ablation, test) không bị chờ vô ích
        time.sleep(1.0)
        still = []
        for proc, item, gpu, t0 in running:
            if proc.poll() is None:
                still.append((proc, item, gpu, t0))
                continue
            mins = (time.time() - t0) / 60.0
            print("[%s] xong GPU %d %s (%.1f phút, exit %d)"
                  % (label, gpu, item, mins, proc.returncode), flush=True)
            if proc.returncode != 0:
                failed.append(item)
        running = still
    return failed


def base_env(gpu=0, **extra):
    """Môi trường cho một tiến trình con: đúng python + đúng GPU."""
    env = dict(os.environ, PYTHON=python(), TOKENIZERS_PARALLELISM="false",
               CUDA_VISIBLE_DEVICES=str(gpu))
    env.update({k: str(v) for k, v in extra.items()})
    return env


def done(tap, seed):
    """Một run coi là xong khi có checkpoint final/ VÀ dump dự đoán.

    Dump do chính main.py ghi trong lúc suy luận (trước bước export), nên run bị
    ngắt ở bước export sẽ không bị train lại — chỉ cần chạy lại để export.
    """
    data = DATASETS[tap]
    run = ROOT / "results" / "runs" / "mvp" / data / ("seed%d" % seed)
    dump = run / "predictions" / ("aste_%s_test_cd_path5.jsonl" % data)
    return (run / "final").is_dir() and dump.exists()


def plan(datasets):
    todo = [(t, s) for t in datasets for s in SEEDS if not done(t, s)]
    mins = sum(EST_TRAIN[t] for t, _ in todo)
    return todo, mins


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", default=None,
                    help="chỉ chạy tập này (lặp được): 14lap, 14res, 15res, 16res")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--ks", default="1,2,3",
                    help="các K cho ablation (K=5 dùng chính kết quả train)")
    ap.add_argument("--num-path", type=int, default=5,
                    help="số góc nhìn K khi train + suy luận (paper dùng 5)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="số GPU chạy song song (mỗi GPU một tiến trình; 2 cho T4 x2)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--skip-dmaste", action="store_true")
    ap.add_argument("--dmaste-checkpoint", default="laptop14/seed0",
                    help="checkpoint nguồn cho zero-shot: <data>/seed<n>")
    ap.add_argument("--dmaste-domains", default=",".join(DMASTE_DOMAINS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    global SEEDS
    SEEDS = [int(x) for x in args.seeds.split(",") if x.strip()]
    datasets = args.only or list(DATASETS)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    env = dict(os.environ, PYTHON=python(), TOKENIZERS_PARALLELISM="false")
    if args.num_path != 5:
        # run.sh already defaults to 5; only export when it differs so that the
        # official K=5 files are not overwritten by a different setting
        env["NUM_PATH"] = str(args.num_path)

    todo, mins = plan(datasets if not args.skip_train else [])
    n_gpu = max(1, min(args.parallel, count_gpus()))
    if args.parallel > 1 and n_gpu < args.parallel:
        print("!! chỉ thấy %d GPU, giảm --parallel xuống %d" % (n_gpu, n_gpu))
    print("=" * 68)
    print("MvP — toàn bộ phần của C")
    print("  python     : %s" % python())
    print("  tập        : %s" % ", ".join(datasets))
    print("  seed       : %s" % SEEDS)
    print("  ablation K : %s" % ks)
    print("  số góc nhìn: K=%d" % args.num_path)
    print("  GPU        : %d (song song %d tiến trình)" % (count_gpus(), n_gpu))
    print("  DMASTE     : %s" % ("(bỏ qua)" if args.skip_dmaste
                                 else args.dmaste_domains))
    print("  còn phải train: %d/%d run" % (len(todo), len(datasets) * len(SEEDS)))
    if mins:
        print("  ước tính %.0f phút (~%.1f giờ) nếu chạy 1 GPU tuần tự"
              % (mins, mins / 60.0))
        if n_gpu > 1:
            print("  chạy %d GPU song song thì còn khoảng %.0f phút (~%.1f giờ)"
                  % (n_gpu, mins / n_gpu, mins / 60.0 / n_gpu))
    print("=" * 68)

    if args.dry_run:
        for tap, seed in todo:
            print("  sẽ train %s seed%d" % (tap, seed))
        return 0

    # ---------------------------------------------------------------- 1. train
    if not args.skip_train:
        print("\n### 1. Train %d tập × %d seed (%d GPU song song)\n"
              % (len(datasets), len(SEEDS), n_gpu))
        pairs = [(t, s) for t in datasets for s in SEEDS if not done(t, s)]
        for tap, seed in [(t, s) for t in datasets for s in SEEDS if done(t, s)]:
            print("[%s seed%d] đã xong, bỏ qua" % (tap, seed))

        failed = run_batch(
            lambda item, gpu: ["bash", "models/multi-view-prompting/run.sh",
                               item[0], str(item[1])],
            pairs, n_gpu, "train")
        if failed:
            print("!! các run sau thất bại: %s"
                  % ", ".join("%s/seed%d" % (t, s) for t, s in failed))
            print("   xem results/logs/mvp/<tập>/seed<n>.log")
            return 1

        # gộp metrics.json -> bảng điểm (cùng định nghĩa với các mô hình khác)
        sh([env["PYTHON"], "models/multi-view-prompting/export_preds.py",
            "--runs", "results/runs/mvp", "--out", "."], env=env)
        sh([env["PYTHON"], "eval/aste_eval.py", "--all"], env=env)

    # ------------------------------------------------------------- 2. ablation
    if not args.skip_ablation and ks:
        print("\n### 2. Ablation số góc nhìn K=%s (dùng checkpoint seed 0)\n" % ks)
        for k in ks:
            jobs = [(tap, k) for tap in datasets if done(tap, 0)]
            for tap in datasets:
                if not done(tap, 0):
                    print("[%s K=%d] thiếu checkpoint seed0, bỏ qua" % (tap, k))
            if not jobs:
                continue
            print("--- K=%d: %d tập" % (k, len(jobs)), flush=True)
            run_batch(
                lambda item, gpu: ["bash", "models/multi-view-prompting/run.sh",
                                   item[0], "0", "ablate"],
                jobs, n_gpu, "K=%d" % k,
                extra_env={"NUM_PATH": k})
            sh([env["PYTHON"], "models/multi-view-prompting/export_preds.py",
                "--runs", "results/runs/mvp", "--out", ".", "--k", str(k)], env=env)
        sh([env["PYTHON"], "models/multi-view-prompting/ablation_table.py",
            "--out", "results/MVP_ABLATION.md"], env=env)

    # --------------------------------------------------------------- 3. DMASTE
    if not args.skip_dmaste:
        print("\n### 3. DMASTE zero-shot\n")
        sh([env["PYTHON"], "models/multi-view-prompting/prepare_dmaste.py",
            "--repo", "models/multi-view-prompting", "--target-only", "--verify"],
           env=env)
        src = ROOT / "results" / "runs" / "mvp" / args.dmaste_checkpoint / "final"
        if not src.is_dir():
            print("!! thiếu checkpoint %s — bỏ qua DMASTE" % src)
        else:
            domains = [d.strip() for d in args.dmaste_domains.split(",") if d.strip()]
            run_batch(
                lambda item, gpu: ["bash", "models/multi-view-prompting/run.sh",
                                   "dmaste", item],
                domains, n_gpu, "dmaste",
                extra_env={"CKPT": str(src), "MAX_SEQ_LENGTH": "256",
                           "EVAL_BATCH_SIZE": "32"})
            sh([env["PYTHON"], "models/multi-view-prompting/eval_dmaste.py",
                "--csv", "results/MVP_DMASTE.csv"], env=env)

    # ------------------------------------------------------------ 4. tổng hợp
    print("\n### 4. Tổng hợp\n")
    sh([env["PYTHON"], "models/multi-view-prompting/error_report.py",
        "--out", "results/MVP_ERRORS.md"], env=env, check=False)
    sh([env["PYTHON"], "models/multi-view-prompting/fill_runs_csv.py",
        "--commit", "chưa commit"], env=env, check=False)

    print("\nXong. Kiểm tra:")
    print("  results/preds/mvp/          dự đoán chính thức (4 tập × 2 seed)")
    print("  results/ablation/           dự đoán theo K")
    print("  results/logs/mvp/           log + .stats.json")
    print("  results/MVP_ERRORS.md       phân tích lỗi")
    print("  results/MVP_ABLATION.md     bảng F1 theo K")
    print("  results/MVP_DMASTE.csv      DMASTE zero-shot")
    print("  results/runs.csv            nhật ký chạy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
