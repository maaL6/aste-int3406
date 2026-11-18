"""Chạy dự đoán trên tập test bằng các checkpoint GTS đã fine-tune, trên GPU T4 của Kaggle,
để đo thời gian suy luận. Không huấn luyện lại."""
import glob, os, subprocess, sys

def sh(cmd, check=True):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check).returncode

sh("nvidia-smi")
sh("find /kaggle/input -maxdepth 3 | head -60")
tar = glob.glob("/kaggle/input/**/aste.tar.gz", recursive=True)
if tar:
    sh("mkdir -p /kaggle/working/aste && tar -xzf {} -C /kaggle/working/aste".format(tar[0]))
else:  # Kaggle đã tự giải nén
    src = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/eval/aste_eval.py", recursive=True)[0]))
    sh("cp -r {} /kaggle/working/aste".format(src))
os.chdir("/kaggle/working/aste")

def ckpt_dir(model):
    found = glob.glob("/kaggle/input/**/aste-ckpt-{}/**/14lap-seed0.pt".format(model), recursive=True)
    return os.path.dirname(found[0]) if found else None

def run_all():
    ok = True
    for model in ("gts-bert", "gts-bilstm"):
        if ckpt_dir(model) is None:
            print("không có checkpoint của", model, "- bỏ qua", flush=True)
            continue
        for tap in ("14lap", "14res", "15res", "16res"):
            for seed in (0, 1):
                rc = sh("OUT=/kaggle/working/results CKPT_DIR={} bash models/GTS/run.sh {} {} {} test".format(
                    ckpt_dir(model), model, tap, seed), check=False)
                if rc != 0:
                    sh("tail -n 15 /kaggle/working/results/logs/{}/{}/seed{}.log".format(model, tap, seed), check=False)
                    return False
    return ok

# Checkpoint lưu cả đối tượng mô hình bằng torch 2.15 và transformers 5.23.
sh("pip install -q transformers==5.23.0")
if not run_all():
    print("Nạp checkpoint lỗi với torch có sẵn, cài torch 2.15.0 rồi chạy lại", flush=True)
    sh("pip install -q torch==2.15.0")
    assert run_all()
sh("python eval/aste_eval.py --all --preds-dir /kaggle/working/results/preds")
sh("cd /kaggle/working && rm -rf aste && find results -name '*.stats.json' | xargs cat")
