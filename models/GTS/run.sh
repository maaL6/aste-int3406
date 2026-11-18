#!/bin/bash
# Nhóm thêm: huấn luyện và dự đoán GTS trên một tập với một seed, rồi ghi kết quả vào results/.
#   bash models/GTS/run.sh gts-bert 14lap 0
#   bash models/GTS/run.sh gts-bilstm 16res 1
# Chỉ dự đoán trên test bằng checkpoint đã có (file $CKPT_DIR/{tập}-seed{n}.pt):
#   CKPT_DIR=/kaggle/input/aste-ckpt-gts-bert bash models/GTS/run.sh gts-bert 14lap 0 test
# Chạy được từ bất kỳ thư mục nào. Biến môi trường:
#   CUDA_VISIBLE_DEVICES  chọn GPU
#   PYTHON                lệnh python (mặc định: python)
#   OUT                   thư mục ghi kết quả (mặc định: results/ của repo)
set -euo pipefail

MODEL=$1          # gts-bert | gts-bilstm
TAP=$2            # 14lap | 14res | 15res | 16res
SEED=$3
MODE=${4:-train}  # train | test

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=${PYTHON:-python}
OUT=${OUT:-$ROOT/results}

case $MODEL in
  gts-bert)   CODE=BertModel; CKPT_PREFIX=bert ;;
  gts-bilstm) CODE=NNModel;   CKPT_PREFIX=bilstm ;;
  *) echo "mô hình không hợp lệ: $MODEL" >&2; exit 1 ;;
esac
# GTS đặt tên tập ngược với nhóm: 14lap -> lap14
DATASET=${TAP:2}${TAP:0:2}

LOG=$OUT/logs/$MODEL/$TAP/seed$SEED.log
PRED=$OUT/preds/$MODEL/$TAP/seed$SEED.jsonl
mkdir -p "$(dirname "$LOG")" "$(dirname "$PRED")"

cd "$ROOT/models/GTS/code/$CODE"
if [ "$MODE" = test ]; then
  # main.py tìm checkpoint theo tên riêng của nó trong savemodel/
  mkdir -p savemodel
  ln -sf "$CKPT_DIR/$TAP-seed$SEED.pt" "savemodel/${CKPT_PREFIX}triplet_${DATASET}_seed$SEED.pt"
fi
{
  echo "commit: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
  $PYTHON -c "import sys, torch, transformers; print('python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'gpu', torch.cuda.get_device_name(0))"
  # tắt thanh tiến độ của tqdm để log gọn
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 $PYTHON main.py --task triplet --mode "$MODE" --dataset "$DATASET" --seed "$SEED" \
    --pred_file "$PRED" --stats_file "${LOG%.log}.stats.json" 2>&1
} > "$LOG"

$PYTHON "$ROOT/eval/aste_eval.py" "$PRED"
