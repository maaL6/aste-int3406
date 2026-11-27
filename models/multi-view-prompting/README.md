# MvP (multi-view prompting) — phần của C

Mã gốc của tác giả nằm ở commit `fd5f5cf`; mọi thay đổi của nhóm là commit riêng,
message bắt đầu bằng `mvp:`. Paper: Gou et al., *MvP: Multi-view Prompting Improves
Aspect Sentiment Tuple Prediction*, ACL 2023. Repo:
<https://github.com/ZubinGou/multi-view-prompting>, commit `549a2b7`.

## Chạy

Repo đã public nên clone không cần token. Trên Kaggle chạy `notebooks/mvp.ipynb`
(Python 3.8 venv dựng ở ô 2), hoặc bằng tay:

```bash
# từng lần chạy (notebook gọi đúng lệnh này)
bash models/multi-view-prompting/scripts/run_main.sh laptop14 0
bash models/multi-view-prompting/scripts/run_main.sh rest16 1

# không tham số = cả 4 tập × 2 seed
PARALLEL=2 bash models/multi-view-prompting/scripts/run_main.sh

# suy luận lại với K khác, dùng checkpoint đã train (không train lại)
MODE=infer NUM_PATH=3 SAVE_DIR_TAG=/k3 \
CKPT=results/runs/mvp/laptop14/seed0/final \
  bash models/multi-view-prompting/scripts/run_main.sh laptop14 0

# DMASTE zero-shot
MODE=infer DATA_OVERRIDE=dmaste_book MAX_SEQ_LENGTH=256 EVAL_BATCH_SIZE=32 \
CKPT=results/runs/mvp/laptop14/seed0/final \
  bash models/multi-view-prompting/scripts/run_main.sh dmaste_book 0

# demo (MvP chạy trực tiếp, ba mô hình còn lại hiện dự đoán lưu sẵn)
python demo/app.py --share
```

## 1. Ý tưởng chính

MvP đưa ASTE về dạng sinh văn bản. Mỗi bộ ba được viết thành chuỗi có đánh dấu phần
tử, ví dụ `[A] battery life [O] long [S] great`, nhiều bộ ba nối bằng `[SSEP]`. Thứ tự
ba phần tử (A, O, S) được hoán vị; mỗi hoán vị là một **góc nhìn**, được nối vào cuối
câu như một prompt. Khi huấn luyện, mỗi câu được nhân thành nhiều bản, mỗi bản một thứ
tự; thứ tự dùng để train được xếp hạng sẵn theo entropy trên tập train (repo chép sẵn
bảng `optim_orders_all`, không tính lại). Khi suy luận, mô hình sinh kết quả cho từng
góc nhìn rồi **bỏ phiếu**: bộ ba nào xuất hiện ở ít nhất một nửa số góc nhìn mới được
giữ.

Hai cơ chế phụ quan trọng: **giải mã có ràng buộc** (`prefix_allowed_tokens_fn`), chỉ
cho phép sinh token hợp lệ theo ngữ pháp `[A] … [O] … [S] …` và chỉ cho phép span lấy
từ chính câu đầu vào; và **bỏ phiếu**, làm tăng precision vì bộ ba chỉ một góc nhìn
nghĩ ra sẽ bị loại.

## 2. Thiết lập chạy và những chỗ nhóm đã sửa

Siêu tham số theo bảng mục 2 của `docs/quy-uoc-chung.md`: `t5-base`, 20 epoch, batch
16, lr 1e-4, chọn checkpoint epoch cuối, 5 góc nhìn, bỏ phiếu. Không dò siêu tham số.

Mọi thay đổi đều có nhãn `MvP-ASTE-TOOLKIT` trong mã, nên `git diff fd5f5cf --
models/multi-view-prompting` là danh sách đầy đủ.

| # | Sửa gì | Vì sao |
|---|---|---|
| 1 | `evaluate()` ghi một dòng JSONL cho mỗi câu test (`predictions/`) | Repo gốc chỉ in F1; cần file dự đoán để chấm bằng script chung và phân tích lỗi |
| 2 | `--stats_file` ghi `phut_train`, `trieu_tham_so`, `ms_moi_cau`, `mb_gpu` | Bảng chi phí của nhóm cần bốn số này, đo cùng định nghĩa với GTS |
| 3 | Pad theo batch (`aste_collate`) thay vì pad mọi câu lên 200/1024 token | Nhanh hơn ~3–4 lần, phép toán không đổi: vị trí pad bị đặt `-100` nên không vào loss, `attention_mask` chặn ở encoder/decoder. Đã kiểm chứng loss trùng tới 1e-4 (`check_fast_padding.py`) |
| 4 | `force_tokens.json` sinh lại khi thiếu, và giữ dict theo task | Bản gốc ghi `dic['sentiment_tokens'] = sp_tokenize_res` đè lên dict vừa dựng, đồng thời dùng lại biến `task` của vòng lặp nên sai task ở lần gọi sau |
| 5 | Bỏ phiếu giữ khía cạnh ẩn; giải mã ràng buộc cho phép token `it` | ASTE-Data-V2 không có khía cạnh ẩn nên hành vi trên bốn tập không đổi; DMASTE thì có |
| 6 | Bỏ checkpoint trung gian (`--save_top_k 0`), chỉ giữ `final/` | Tiết kiệm đĩa; quy ước chọn checkpoint của MvP là epoch cuối, không phải F1 dev cao nhất |
| 7 | Sửa `scripts/run_main.sh` cho chạy đủ 4 tập × 2 seed, ghi `results/`, và thêm `MODE=infer` để suy luận lại không train | Đúng yêu cầu mục 3 của `docs/phan-cong-cong-viec.md`. Giữ nguyên cách repo gốc tách `run_main.sh` (có `--do_train`) và `eval_unified.sh` (không có) |

Chạy trên **2 GPU song song**, mỗi run một GPU riêng (không DDP). Kaggle tính 1 giờ
trên 2 GPU là 1 giờ quota nên song song chỉ lãi thời gian; kết quả giống hệt chạy tuần
tự vì mỗi run độc lập. Vòng lặp nằm trong notebook (`PARALLEL = 2`): xong run nào thì
nhận run tiếp theo, mỗi tiến trình nhận `CUDA_VISIBLE_DEVICES=0` hoặc `1` (vì `main.py`
luôn gọi `cuda:0`, đặt biến này là cách duy nhất để nó thấy đúng card).

## 3. Kết quả so với paper và lý do chênh lệch

Số của nhóm lấy từ `eval/aste_eval.py` (so trên văn bản), không lấy số repo in ra:

| Tập | seed 0 | seed 1 | Nhóm (mean ± nửa khoảng) | Paper (5 seed) | Chênh |
|---|---:|---:|---:|---:|---:|
| 14lap | 63.00 | 63.18 | **63.09 ± 0.09** | 63.33 | −0.24 |
| 14res | 74.75 | 73.91 | **74.33 ± 0.42** | 74.05 | +0.28 |
| 15res | 66.33 | 66.46 | **66.40 ± 0.07** | 65.89 | +0.51 |
| 16res | 74.90 | 74.24 | **74.57 ± 0.33** | 73.48 | +1.09 |

Ba tập cao hơn paper, 14lap thấp hơn 0.24. Bốn nguyên nhân, xếp theo mức ảnh hưởng:

1. **Khác cách so khớp.** Nhóm so trên văn bản đã chuẩn hóa; paper so khớp tuyệt đối
   trên chuỗi sinh. Với 14lap, script của nhóm chỉ tính 541 bộ ba ở tập test thay vì
   543 (hai bộ ba bị ghi lặp trong file gốc), nên recall lệch nhẹ so với paper.
2. **Khác số seed.** Paper báo trung bình 5 seed, nhóm chạy 2. Khoảng cách giữa hai
   seed của nhóm tới 0.84 điểm (14res), nên chênh lệch dưới 0.5 điểm không kết luận
   được gì. Ở 16res cả hai seed đều cao hơn số paper nên đây là khác biệt thật, nhiều
   khả năng do phiên bản `transformers`/`torch` khác bản tác giả dùng.
3. **Khác cách tính recall ở 14lap.** Như trên: mẫu số 541 thay vì 543.
4. **Pad theo batch.** Đã kiểm chứng loss trùng tới 1e-4 nên không phải nguồn lệch.

**Kiểm chứng định lượng cho khoảng cách giữa hai bộ số.** `result.txt` của seed 0 có
bốn dòng, theo thứ tự K=5, K=1, K=2, K=3 (lần chạy K=5 sinh ra file, ba lần ablation
ghi tiếp vào). Đối chiếu với bảng ablation ở mục 6:

| Tập | Repo K=1 | Nhóm K=1 | Repo K=2 | Nhóm K=2 | Repo K=3 | Nhóm K=3 |
|---|---:|---:|---:|---:|---:|---:|
| 14lap | 63.48 | 63.30 | 61.75 | 61.58 | 63.54 | 63.36 |
| 14res | 74.03 | 73.83 | 73.28 | 72.99 | 74.70 | 74.50 |
| 15res | 65.28 | 64.89 | 63.93 | 63.56 | 66.33 | 65.93 |
| 16res | 73.73 | 73.62 | 73.76 | 73.65 | 74.60 | 74.41 |

Chênh lệch nằm trong khoảng −0.10 đến −0.40 và **luôn cùng dấu** (nhóm thấp hơn) ở cả
12 ô. Nếu là nhiễu thì dấu phải đảo. Đó là bằng chứng rằng khoảng cách này do cách so
chuỗi, không phải do mô hình hay may mắn. Nguyên nhân cụ thể ở mục 4 dưới đây.

## 4. File dự đoán và lỗi do detokenize

`results/preds/mvp/{tập}/seed{n}.jsonl` theo mục 4 của quy ước. MvP là mô hình sinh
văn bản nên `aspect` và `opinion` là `null`, chỉ có `aspect_text` và `opinion_text`.

Một điểm phải xử lý riêng: MvP giải mã ra chữ đã nối (`wasn't`, `jukebox's`), còn câu
gốc tách token (`was n't`, `juke box 's`). So chuỗi thô thì mọi bộ ba chứa dấu nháy đều
bị tính sai dù dự đoán đúng. `export_preds.py` vì vậy căn span về đúng token của câu
trước khi ghi, nên hai phía của phép so lấy từ cùng một danh sách token. Ảnh hưởng đo
được trên dump thật: 14res +0.10 F1, 16res +0.57, 14lap và 15res không đổi.

## 5. Lỗi điển hình và giải thích

Chi tiết ở `results/MVP_ERRORS.md` (sinh bằng
`python models/multi-view-prompting/error_report.py`; nhãn loại lỗi theo mục 10 của
quy ước). Tóm tắt 14lap seed 0 — 541 bộ ba vàng, dự đoán 513, đúng 332:

| Loại lỗi | Số | % lỗi phía đáp án |
|---|---:|---:|
| Sai biên | 85 | 40.7% |
| Bỏ sót | 77 | 36.8% |
| Sai cảm xúc | 35 | 16.7% |
| Sai cặp | 12 | 5.7% |
| Dự đoán thừa | 49 | — |
| Sai định dạng | 0 | — |

Bốn nhận xét rút ra từ cơ chế mô hình:

- **Sai biên là loại phổ biến nhất.** MvP chọn span bằng cách sinh token từ câu, nhưng
  không có gì buộc nó dừng đúng chỗ: `risottos ( particularly the sepia )` thay vì chỉ
  `risottos`. Đây là hệ quả trực tiếp của việc dùng T5 như bộ sinh tự do, khác
  Span-ASTE vốn phân loại từng span ứng viên nên biên chính xác hơn.
- **Bỏ sót đứng thứ hai và tập trung ở câu nhiều bộ ba.** Câu có 5 bộ ba chỉ đạt F1
  12.5% (14lap): chuỗi `[SSEP]` dài dễ bị cắt hoặc sinh thiếu, và bỏ phiếu loại những
  bộ ba chỉ 1–2 góc nhìn nghĩ ra — mà câu nhiều bộ ba đúng là nơi các góc nhìn bất đồng
  nhiều nhất.
- **Sai cảm xúc (16.7%)** chủ yếu là lưỡng lự giữa `ok` và `great`/`bad` (`adequate`
  → POS thay vì NEU). MvP chỉ sinh một trong ba token `great/ok/bad` nên mất thông tin
  về mức độ; mô hình lưới phân loại trên biểu diễn của cặp span nên tách NEU tốt hơn.
- **Sai định dạng bằng 0 ở cả 8 lần chạy.** Giải mã có ràng buộc làm đúng việc của nó:
  mọi chuỗi sinh ra đều đọc được thành bộ ba. Đây là điểm mạnh so với mô hình sinh
  không ràng buộc, và là lý do dự đoán "MvP dễ lỗi định dạng" trong kế hoạch không
  đúng với cấu hình có `--constrained_decode`.

## 6. Thí nghiệm số góc nhìn K

Dùng lại checkpoint seed 0, chỉ đổi `--num_path` lúc suy luận nên chỉ tốn thời gian
sinh. F1 chấm bằng `eval/aste_eval.py`:

| Tập | K=1 | K=2 | K=3 | K=5 |
|---|---:|---:|---:|---:|
| 14lap | 63.30 | 61.58 | 63.36 | 63.00 |
| 14res | 73.83 | 72.99 | 74.50 | 74.75 |
| 15res | 64.89 | 63.56 | 65.93 | 66.33 |
| 16res | 73.62 | 73.65 | 74.41 | 74.90 |
| **Trung bình** | **68.91** | **67.94** | **69.55** | **69.75** |

Sinh lại bằng `python models/multi-view-prompting/ablation_table.py`. Ba điểm đáng nói:

- **K=2 là điểm xấu nhất, tệ hơn cả K=1.** Ngưỡng bỏ phiếu là `count >= K/2`, với K=2
  thành `>= 1`, tức một góc nhìn nói gì cũng được giữ — bước bỏ phiếu mất tác dụng
  lọc. P tụt mạnh (14res 69.65 so với 72.37 ở K=1) trong khi R tăng. Đây là lỗi thiết
  kế ngưỡng của repo gốc, không phải mô hình yếu đi.
- Từ K=3 trở lên ngưỡng thành `>= 1.5` (làm tròn lên 2) nên bỏ phiếu có tác dụng thật;
  F1 tăng đơn điệu tới K=5.
- 14lap là tập duy nhất K=1 và K=3 nhỉnh hơn K=5, nhưng chênh chỉ 0.3–0.36 điểm trên
  328 câu test nên không kết luận được.

## 7. Số đo chi phí và dev F1

`results/logs/mvp/{tập}/seed{n}.log` là log huấn luyện thật của session Kaggle,
`seed{n}.result.txt` là `result.txt` của repo, `seed{n}.stats.json` là các số đã trích.

| Tập | seed | dev F1 | phút train | ms/câu | MB GPU | repo F1 |
|---|---|---:|---:|---:|---:|---:|
| 14lap | 0 | 60.80 | 53.2 | 544.1 | 7888 | 63.19 |
| 14lap | 1 | 59.40 | 53.3 | 538.3 | 8003 | 63.56 |
| 14res | 0 | 64.50 | 77.3 | 717.2 | 8836 | 74.95 |
| 14res | 1 | 63.40 | 77.2 | 754.0 | 8856 | 74.11 |
| 15res | 0 | 71.80 | 34.7 | 541.9 | 8819 | 66.73 |
| 15res | 1 | 73.60 | 34.8 | 531.8 | 8830 | 66.87 |
| 16res | 0 | 72.90 | 48.8 | 484.0 | 8338 | 75.10 |
| 16res | 1 | 74.00 | 48.8 | 489.3 | 8402 | 74.81 |

Bốn điều phải nói rõ trong báo cáo, nếu không người đọc sẽ tưởng bảng sai:

- **`dev_f1` không so được với `test_f1`.** Hàm `evaluate` trong `T5FineTuner` chấm dev
  bằng giải mã **không ràng buộc, 1 góc nhìn, không bỏ phiếu**, còn test mới chạy đủ
  công thức paper (ràng buộc + 5 góc nhìn + bỏ phiếu). Vì vậy 14res có dev 64.5 mà
  test 74.95, còn 15res ngược lại (dev 71.8, test 66.3 — dev của 15res dễ hơn test).
  Chỉ số dev dùng để vẽ đường hội tụ và chứng minh mô hình có học, **không** dùng để
  chọn checkpoint (MvP giữ epoch cuối).
- **`ms_moi_cau` là cho cả 5 góc nhìn**, không phải một lượt sinh. Chia cho K nếu bảng
  chi phí cần "mỗi góc nhìn" (≈ 97–151 ms). Đo trên T4, chỉ tính phần `generate` của
  tập test, không tính nạp mô hình và tokenize.
- **`mb_gpu` là `torch.cuda.max_memory_allocated()`** đo sau khi train xong, đúng định
  nghĩa mục 6. Dao động 7.9–8.9 GB tuỳ độ dài câu của từng tập (14res dài nhất). Con số
  này đo trên T4 15 GB với batch 16 × 200 token; nếu chạy batch lớn hơn thì phải đo lại.
- **Thời gian và bộ nhớ đo trên T4, chạy 2 GPU song song.** Mỗi run một GPU riêng nên
  không tranh VRAM, nhưng `phut_train` có thể cao hơn chạy đơn lẻ ~5% do chia sẻ CPU
  (tokenize dùng 2 worker). Đừng so trực tiếp với bảng của GTS nếu GTS chạy tuần tự.

## 8. Demo Gradio

`demo/app.py`. Hai mô hình MvP chạy **trực tiếp** từ checkpoint; năm mô hình còn lại
(`span-aste`, `gts-bert`, `emcgcn`, `rule`, `crf`) hiện **dự đoán đã lưu** trong
`results/preds/`, nên chỉ trả lời được câu có trong tập test. Cách này tránh phải nạp
bốn bộ thư viện khác nhau vào cùng một tiến trình (đúng như `phan-cong-cong-viec.md`
mục 6 dự phòng).

```bash
python demo/app.py --share                     # mở link *.gradio.live
python demo/app.py --self-test                 # kiểm tra backend, không cần Gradio
python demo/app.py --self-test --domain book   # kiểm tra nhánh DMASTE
python demo/app.py --no-mvp                    # chỉ hiện dự đoán lưu sẵn
```

Trong UI có dropdown **Domain**: `aste` dùng checkpoint ASTE-Data-V2 với
`max_seq_length 200`; `book`/`pet`/`toy`/`grocery` dùng **cùng checkpoint đó** nhưng
chạy zero-shot với `max_seq_length 256`, đúng cấu hình lúc chấm điểm.

Ba điểm cần biết khi chạy trên Kaggle:

- **Cần GPU** cho phần MvP trực tiếp. Không có GPU thì `demo/app.py` vẫn mở và hiện dự
  đoán lưu sẵn, chỉ hai dòng MvP báo "không có checkpoint để nạp".
- **Checkpoint không nằm trong git** (`.gitignore` chặn `*.bin`). Phải Add Input
  Kaggle Dataset `aste-ckpt-mvp` hoặc output session đã train; `discover_checkpoints`
  tự tìm trong `/kaggle/input` nếu `results/runs/` rỗng.
- **`force_tokens.json`** phải khớp task đang chạy. Script tự tạo file này nếu thiếu,
  nhưng nếu file cũ còn sót từ lần chạy DMASTE thì chuỗi `[A] x [O] y [S] z` vẫn đúng
  nên không ảnh hưởng.

## 9. DMASTE zero-shot

Theo kế hoạch nhóm (mục 5): dùng checkpoint đã train trên ASTE-Data-V2, chạy thẳng lên
bốn domain đích (`book`, `pet`, `toy`, `grocery` — chỉ có dev/test), không train lại.
Bỏ các bộ ba có khía cạnh ẩn vì MvP không được huấn luyện để sinh ra chúng, và kế
hoạch cũng ghi "nếu có thì lọc".

```bash
python models/multi-view-prompting/prepare_dmaste.py --target-only --verify
MODE=infer DATA_OVERRIDE=dmaste_book \
MAX_SEQ_LENGTH=256 EVAL_BATCH_SIZE=32 \
CKPT=results/runs/mvp/laptop14/seed0/final \
  bash models/multi-view-prompting/scripts/run_main.sh dmaste_book 0
python models/multi-view-prompting/eval_dmaste.py --csv results/MVP_DMASTE.csv
```

Hai tham số phải đổi so với ASTE-Data-V2, nếu không kết quả sẽ sai:

- **`MAX_SEQ_LENGTH=256`.** Câu DMASTE + prompt dài tới 242 token; để mặc định 200
  thì truncation cắt mất span trong khi chỉ số span vẫn tính trên câu gốc.
- **`EVAL_BATCH_SIZE=32`.** 256 token × batch 64 dễ OOM trên T4 15 GB.

### Kết quả

| Domain | P | R | F1 (nhóm) | F1 (repo) | ms/câu | MB GPU |
|---|---:|---:|---:|---:|---:|---:|
| book | 22.37 | 15.08 | **18.02** | 17.47 | 710 | 2659 |
| pet | 28.74 | 17.24 | **21.55** | 19.92 | 630 | 2924 |
| toy | 24.85 | 14.91 | **18.64** | 18.43 | 637 | 2793 |
| grocery | 28.60 | 16.86 | **21.22** | 20.67 | 595 | 2652 |
| **micro** | **25.96** | **15.96** | **19.77** | — | | |

### Bốn điều phải nói rõ trong báo cáo

**1. Đây không so được với bảng DMASTE của paper.** Paper báo số **in-domain**
(train riêng từng domain, giữ khía cạnh ẩn: Electronics 47.86 / Beauty 46.46 /
Fashion 50.38 / Home 49.14). Nhóm chạy **zero-shot** trên **4 domain khác** và **bỏ
hơn 40% bộ ba** (khía cạnh ẩn). Ba khác biệt cùng lúc, nên chỉ dùng cột Span-ASTE của
A làm mốc nếu A cũng chạy zero-shot cùng 4 domain.

**2. Recall thấp là nguyên nhân chính, không phải precision.** Micro P = 25.96 nhưng
R = 15.96. Model dự đoán 1,737 bộ ba trong khi đáp án có 2,826 — thiếu 39%.

**3. Recall giảm đều theo độ dài câu**, đây là bằng chứng định lượng cho giả thuyết
"train trên câu ngắn, test trên câu dài":

| Độ dài câu | book | pet | toy | grocery |
|---|---:|---:|---:|---:|
| 21–40 từ | 21.0% | 26.7% | 26.5% | 26.0% |
| 41–80 từ | 15.1% | 20.0% | 15.0% | 13.7% |
| > 80 từ | 9.7% | 8.4% | 7.8% | 8.9% |

Recall giảm khoảng 3 lần từ câu ngắn sang câu dài, ở cả 4 domain. MvP học từ câu
nhà hàng/laptop trung bình 19–20 từ với 1.8 bộ ba; DMASTE trung bình 60–70 từ với
3.3 bộ ba, nhiều câu trên 200 từ.

**4. Truncation chỉ ảnh hưởng 1–2% số câu**, không phải nguyên nhân chính. Kiểm tra:
chỉ 3/260 câu của book dài hơn 200 token, và những câu đó đúng là câu sinh ra bộ ba
rỗng. Nhưng phần lớn tổn thất nằm ở câu 40–80 từ, vốn không bị cắt.

### Vì sao số của nhóm khác số của repo

Chênh lệch không phải do cách so khớp mà do **bộ ba rỗng**. Với `pet`:

| | gold | pred | hit | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|
| repo | 580 | **424** | 100 | 23.58 | 17.24 | 19.92 |
| nhóm | 580 | **348** | 100 | 28.74 | 17.24 | **21.55** |

`hit` và `recall` **giống hệt**, chỉ `pred` khác: 424 − 348 = 76 bộ ba rỗng
(`[A] null [O] null [S] ...`) mà repo đếm vào mẫu số precision. Chúng chắc chắn không
khớp đáp án, nên giữ lại chỉ làm precision thấp giả tạo. Recall không đổi vì không có
bộ ba rỗng nào trong đáp án.

Đây là lý do phải lọc chúng, và cũng là lý do khi báo cáo nên dùng số của nhóm
(`test_f1`) làm số chính, số repo chỉ để đối chiếu.

Số liệu đã điền vào `results/runs.csv` (4 dòng `mvp-dmaste`), cột `ghi_chu` ghi rõ
"zero-shot, bỏ mẫu khía cạnh ẩn".
