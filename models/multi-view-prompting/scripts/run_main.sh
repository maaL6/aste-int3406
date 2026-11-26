set -ex

# Nhóm sửa (docs/phan-cong-cong-viec.md mục 3): chạy đủ 4 tập của ASTE-Data-V2 và ghi
# kết quả vào results/ của repo nhóm theo docs/quy-uoc-chung.md.
#
#   bash scripts/run_main.sh                    # 4 tập × 2 seed, train + dự đoán
#   bash scripts/run_main.sh laptop14 0         # một tập, một seed
#   NUM_PATH=3 bash scripts/run_main.sh         # đổi số góc nhìn K
#   PARALLEL=2 bash scripts/run_main.sh         # 2 run cùng lúc (Kaggle T4 x2)
#   MODE=infer bash scripts/run_main.sh ...     # chỉ suy luận lại, không train
#
# MODE theo đúng cách repo gốc tách hai việc: `run_main.sh` có `--do_train`,
# `eval_unified.sh` không có (chỉ suy luận từ checkpoint đã lưu).
#
# Biến môi trường:
#   MODE                  train (mặc định) | infer
#   DATA_OVERRIDE         tên tập trong data/aste (mặc định suy ra từ tham số)
#   CKPT                  với MODE=infer, dùng final/ ở đây thay vì train mới
#   CUDA_VISIBLE_DEVICES  GPU (PARALLEL > 1 thì script tự chia 0,1,...)
#   PYTHON                lệnh python (mặc định: python)
#   OUT                   thư mục gốc ghi kết quả (mặc định: results/ của repo)
#   NUM_PATH              số góc nhìn K (mặc định 5, paper dùng 5)
#   PARALLEL              số run chạy cùng lúc (mặc định 1)
#   MAX_SEQ_LENGTH        mặc định 200; DMASTE cần 256
#   EVAL_BATCH_SIZE       mặc định 64; DMASTE để 32 cho khỏi OOM trên T4
#   SAVE_DIR_TAG          thêm hậu tố vào thư mục run (dùng cho ablation theo K)

TASK=aste
ALL_DATA="laptop14 rest14 rest15 rest16"   # tên tập theo repo MvP

PYTHON=${PYTHON:-python}
# script nằm ở models/multi-view-prompting/scripts/ -> MVP là gốc repo MvP,
# ROOT là gốc repo nhóm (hai cấp trên MVP)
MVP=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(cd "$MVP/../.." && pwd)
OUT=${OUT:-$ROOT/results}

# báo sớm nếu PYTHON không chạy được, thay vì để 8 job cùng fail.
# Chỉ kiểm tra khi PYTHON là tên lệnh (python, python3...); nếu là đường dẫn thì
# để chính lệnh gọi báo lỗi, vì `command -v` xử lý đường dẫn khác nhau giữa các shell.
case "$PYTHON" in
  */*|*\\*) : ;;
  *) command -v "$PYTHON" >/dev/null 2>&1 || {
       echo "PYTHON='$PYTHON' không chạy được. Đặt PYTHON=/đường/dẫn/python" >&2
       exit 1; } ;;
esac

NUM_PATH=${NUM_PATH:-5}
PARALLEL=${PARALLEL:-1}
MODE=${MODE:-train}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-200}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-64}
SAVE_DIR_TAG=${SAVE_DIR_TAG:-}
K=$NUM_PATH
CTRL_TOKEN=post
DATA_RATIO=1.0

cd "$MVP/src"

# tập trong data/aste: tên repo MvP, hoặc DATA_OVERRIDE (vd dmaste_book)
to_data_name() {
  if [ -n "${DATA_OVERRIDE:-}" ]; then
    echo "$DATA_OVERRIDE"
    return
  fi
  case $1 in
    laptop14|14lap) echo laptop14 ;;
    rest14|14res)   echo rest14 ;;
    rest15|15res)   echo rest15 ;;
    rest16|16res)   echo rest16 ;;
    *)              echo "$1" ;;
  esac
}

run_one() {
  TAP=$1
  SEED=$2
  GPU=$3
  DATA=$(to_data_name "$TAP")
  LOG=$OUT/logs/mvp/$TAP/seed$SEED${SAVE_DIR_TAG}.log
  RUN=$OUT/runs/mvp/$DATA/seed$SEED${SAVE_DIR_TAG}
  STATS=${LOG%.log}.stats.json

  mkdir -p "$RUN" "$(dirname "$LOG")"

  if [ "$MODE" = infer ]; then
    # dùng checkpoint có sẵn: symlink final/ thay vì train
    [ -d "${CKPT:-}" ] || { echo "MODE=infer cần CKPT=.../final" >&2; exit 1; }
    ln -sfn "$CKPT" "$RUN/final"
    DO_TRAIN=""
  else
    DO_TRAIN="--do_train"
  fi

  CUDA_VISIBLE_DEVICES=$GPU \
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 $PYTHON main.py \
    --data_path "../data/" \
    --dataset "$DATA" \
    --model_name_or_path t5-base \
    --output_dir "$RUN" \
    --num_train_epochs 20 \
    --save_top_k 0 \
    --task $TASK \
    --top_k $K \
    --ctrl_token $CTRL_TOKEN \
    --multi_path \
    --num_path $NUM_PATH \
    --seed $SEED \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lowercase \
    --sort_label \
    --data_ratio $DATA_RATIO \
    --check_val_every_n_epoch 10 \
    --agg_strategy vote \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --constrained_decode \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --stats_file "$STATS" \
    $DO_TRAIN \
    > "$LOG" 2>&1
}

# danh sách job: tham số dòng lệnh, hoặc mặc định 4 tập × 2 seed
if [ $# -ge 2 ]; then
  JOBS="$1:$2"
else
  JOBS=""
  for DATA in $ALL_DATA; do
    for SEED in 0 1; do
      JOBS="$JOBS $DATA:$SEED"
    done
  done
fi

# chạy theo lô PARALLEL, mỗi job một GPU; lô xong mới sang lô tiếp.
# Job chạy nền nên PHẢI tự ghi lại exit code: nếu main.py chết mà không ai kiểm tra
# thì script vẫn "xong" và bước export chỉ báo MISSING, che mất lỗi thật trong log.
gpu=0
n=0
LAST_SEED=0
FAIL_MARK=$(mktemp)
for job in $JOBS; do
  LAST_SEED=${job##*:}
  TAP=${job%%:*}
  SEED=${job##*:}
  (
    run_one "$TAP" "$SEED" $gpu
    rc=$?
    if [ "$rc" -ne 0 ]; then
      LOG=$OUT/logs/mvp/$TAP/seed$SEED${SAVE_DIR_TAG}.log
      {
        echo "!! $TAP seed$SEED lỗi (exit $rc). 30 dòng cuối của $LOG:"
        tail -n 30 "$LOG" 2>/dev/null
        echo ""
      } >> "$FAIL_MARK"
    fi
    exit $rc
  ) &
  gpu=$(( (gpu + 1) % PARALLEL ))
  n=$(( n + 1 ))
  if [ $(( n % PARALLEL )) -eq 0 ]; then
    wait
  fi
done
wait

if [ -s "$FAIL_MARK" ]; then
  cat "$FAIL_MARK" >&2
  rm -f "$FAIL_MARK"
  echo "!! có run thất bại — dừng trước khi export để không tạo file dự đoán thiếu." >&2
  exit 1
fi
rm -f "$FAIL_MARK"

# ghi file dự đoán đúng format chung.
# Lưu ý: DATA/SEED là biến CỤC BỘ trong run_one(), nên ở đây phải tự suy ra lại;
# LAST_SEED lấy từ vòng lặp trên, DATA suy từ DATA_OVERRIDE.
if [ -n "${DATA_OVERRIDE:-}" ]; then
  # DMASTE: mỗi domain ghi vào preds/mvp-dmaste/<domain>/seed<n>.jsonl
  DOMAIN=${DATA_OVERRIDE#dmaste_}
  DATA=$(to_data_name "$DATA_OVERRIDE")
  "$PYTHON" "$ROOT/models/multi-view-prompting/export_preds.py" \
    --runs "$OUT/runs/mvp/$DATA/seed$LAST_SEED${SAVE_DIR_TAG}" --out "$ROOT" \
    --dmaste --domain "$DOMAIN" --seed "$LAST_SEED"
else
  "$PYTHON" "$ROOT/models/multi-view-prompting/export_preds.py" \
    --runs "$OUT/runs/mvp" --out "$ROOT" --k "$NUM_PATH"
fi