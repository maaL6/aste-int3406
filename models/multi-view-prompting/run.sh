#!/bin/bash
# Nhóm thêm: huấn luyện và dự đoán MvP trên một tập với một seed, rồi ghi kết quả vào results/.
#
# ASTE-Data-V2:
#   bash models/multi-view-prompting/run.sh 14lap 0
#   bash models/multi-view-prompting/run.sh 16res 1
#
# DMASTE zero-shot (kế hoạch nhóm, mục 5): dùng checkpoint ASTE-Data-V2 đã train,
# chạy thẳng lên một domain đích, không train lại.
#   CKPT=/path/to/results/runs/mvp/laptop14/seed0/final \
#     bash models/multi-view-prompting/run.sh dmaste book
#
# Chỉ dự đoán lại trên test bằng checkpoint đã có (Kaggle Dataset aste-ckpt-mvp):
#   CKPT_DIR=/kaggle/input/aste-ckpt-mvp bash models/multi-view-prompting/run.sh 14lap 0 test
#
# Chạy được từ bất kỳ thư mục nào. Biến môi trường:
#   CUDA_VISIBLE_DEVICES  chọn GPU
#   PYTHON                lệnh python (mặc định: python)
#   OUT                   thư mục ghi kết quả (mặc định: results/ của repo)
#   NUM_PATH              số góc nhìn K (mặc định 5, paper dùng 5)
set -euo pipefail

TAP=$1                      # 14lap | 14res | 15res | 16res | dmaste
SEED_OR_DOMAIN=${2:-0}
MODE=${3:-train}            # train | test

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python}
OUT=${OUT:-$ROOT/results}
NUM_PATH=${NUM_PATH:-5}

case $TAP in
  14lap) DATA=laptop14; SEED=$SEED_OR_DOMAIN ;;
  14res) DATA=rest14;   SEED=$SEED_OR_DOMAIN ;;
  15res) DATA=rest15;   SEED=$SEED_OR_DOMAIN ;;
  16res) DATA=rest16;   SEED=$SEED_OR_DOMAIN ;;
  dmaste)
    # $2 là tên domain, chỉ số seed lấy mặc định 0 (zero-shot nên 1 seed là đủ)
    DOMAIN=$SEED_OR_DOMAIN
    DATA=dmaste_$DOMAIN
    SEED=${SEED:-0}
    ;;
  *) echo "tập không hợp lệ: $TAP" >&2; exit 1 ;;
esac

if [ "$TAP" = dmaste ]; then
  LOG=$OUT/logs/mvp/dmaste/$DOMAIN/seed$SEED.log
  RUN=$OUT/runs/mvp/dmaste_$DOMAIN/seed$SEED
  STATS=${LOG%.log}.stats.json
  [ -f "$ROOT/models/multi-view-prompting/data/aste/$DATA/test.txt" ] || {
    echo "thiếu data/aste/$DATA/test.txt — chạy prepare_dmaste.py trước" >&2; exit 1; }
  [ -d "${CKPT:-}" ] || { echo "đặt CKPT=.../final của một checkpoint ASTE-Data-V2" >&2; exit 1; }
  mkdir -p "$(dirname "$LOG")"
  # dùng chung final/ của checkpoint nguồn, không copy (đỡ tốn đĩa)
  mkdir -p "$RUN"
  ln -sfn "$CKPT" "$RUN/final"
  EXTRA="--do_inference True"
  # DMASTE dai hon ASTE: cau + prompt toi 242 token, de 200 se cat mat span.
  # 4 domain in-domain cua paper dung 256; zero-shot cung phai 256.
  MAX_SEQ=${MAX_SEQ_LENGTH:-256}
  # 256 token x batch 64 de OOM tren T4 15 GB; 32 van nhanh va an toan
  EVAL_BATCH=${EVAL_BATCH_SIZE:-32}
else
  LOG=$OUT/logs/mvp/$TAP/seed$SEED.log
  RUN=$OUT/runs/mvp/$DATA/seed$SEED
  STATS=${LOG%.log}.stats.json
  MAX_SEQ=${MAX_SEQ_LENGTH:-200}
  mkdir -p "$RUN" "$(dirname "$LOG")"
  if [ "$MODE" = test ]; then
    [ -d "$CKPT_DIR/$TAP-seed$SEED/final" ] || {
      echo "thiếu $CKPT_DIR/$TAP-seed$SEED/final" >&2; exit 1; }
    RUN=$CKPT_DIR/$TAP-seed$SEED
    EXTRA="--do_inference True"
  elif [ "$MODE" = ablate ]; then
    # dùng lại checkpoint vừa train trong phiên này, chỉ đổi số góc nhìn
    BASE=$OUT/runs/mvp/$DATA/seed$SEED
    [ -d "$BASE/final" ] || {
      echo "thiếu $BASE/final — train run này trước (MODE mặc định)" >&2; exit 1; }
    # mỗi K một thư mục riêng để không ghi đè cache path của K khác
    LOG=$OUT/logs/mvp/$TAP/seed$SEED.k$NUM_PATH.log
    STATS=${LOG%.log}.stats.json
    RUN=$BASE/k$NUM_PATH
    mkdir -p "$RUN"
    ln -sfn "$BASE/final" "$RUN/final"
    EXTRA="--do_inference True"
  else
    EXTRA="--do_train --do_inference True"
  fi
fi

cd "$ROOT/models/multi-view-prompting/src"

{
  echo "commit: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
  $PYTHON -c "import sys, torch, transformers, pytorch_lightning as pl; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'pl', pl.__version__, 'gpu', torch.cuda.get_device_name(0))"
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 $PYTHON main.py \
    --data_path "../data/" \
    --dataset "$DATA" \
    --model_name_or_path t5-base \
    --output_dir "$RUN" \
    --num_train_epochs 20 \
    --save_top_k 0 \
    --task aste \
    --top_k 5 \
    --ctrl_token post \
    --multi_path \
    --num_path "$NUM_PATH" \
    --seed "$SEED" \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lowercase \
    --sort_label \
    --data_ratio 1.0 \
    --check_val_every_n_epoch 10 \
    --agg_strategy vote \
    --eval_batch_size "${EVAL_BATCH:-64}" \
    --constrained_decode \
    --max_seq_length "$MAX_SEQ" \
    --stats_file "$STATS" \
    $EXTRA 2>&1
} > "$LOG"

# xuất file dự đoán đúng format chung, rồi chấm bằng script chung
if [ "$TAP" = dmaste ]; then
  $PYTHON "$ROOT/models/multi-view-prompting/export_preds.py" \
    --runs "$RUN" --out "$ROOT" --dmaste --domain "$DOMAIN" --seed "$SEED"
  $PYTHON "$ROOT/models/multi-view-prompting/eval_dmaste.py" --domain "$DOMAIN"
else
  $PYTHON "$ROOT/models/multi-view-prompting/export_preds.py" \
    --runs "$OUT/runs/mvp" --out "$ROOT" --k "$NUM_PATH"
fi
