# MvP (multi-view prompting) — phần của C

Mã gốc của tác giả nằm ở commit `fd5f5cf`; mọi thay đổi của nhóm là commit riêng,
message bắt đầu bằng `mvp:`. Paper: Gou et al., *MvP: Multi-view Prompting Improves
Aspect Sentiment Tuple Prediction*, ACL 2023. Repo:
<https://github.com/ZubinGou/multi-view-prompting>, commit `549a2b7`.

## Chạy

```bash
# trên Kaggle: notebooks/mvp.ipynb (Python 3.8 + torch 1.12 + transformers 4.14.1)
# Notebook chỉ gọi run_all.py; dùng được cho cả 4 phần:
python models/multi-view-prompting/run_all.py                    # tất cả
python models/multi-view-prompting/run_all.py --parallel 2        # 2 GPU song song
python models/multi-view-prompting/run_all.py --only 14lap --only 14res
python models/multi-view-prompting/run_all.py --skip-ablation --skip-dmaste
python models/multi-view-prompting/run_all.py --dry-run           # in kế hoạch

# từng phần lẻ
bash models/multi-view-prompting/run.sh 14lap 0
bash models/multi-view-prompting/run.sh 14lap 0 ablate      # đổi K, dùng lại final/
CKPT=results/runs/mvp/laptop14/seed0/final \
  bash models/multi-view-prompting/run.sh dmaste book

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
| 7 | Thêm `run_all.py` điều phối cả 4 phần, `run.sh` nhận thêm `MODE=ablate` và nhánh `dmaste` | Notebook chỉ còn lệnh cài/chạy (mục 8 của quy ước); ablation và zero-shot chạy được bằng một lệnh |

Chạy trên **2 GPU song song**, mỗi run một GPU riêng (không DDP). Kaggle tính 1 giờ
trên 2 GPU là 1 giờ quota nên song song chỉ lãi thời gian; kết quả giống hệt chạy tuần
tự vì mỗi run độc lập. `run_all.py --parallel 2` tự chia GPU: xong run nào thì nhận
run tiếp theo, mỗi tiến trình nhận `CUDA_VISIBLE_DEVICES=0` hoặc `1` (vì `main.py`
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

| Tập | seed | dev F1 | phút train | ms/câu | repo F1 |
|---|---|---:|---:|---:|---:|
| 14lap | 0 | 60.80 | 50.9 | 478.7 | 63.19 |
| 14lap | 1 | 59.40 | 50.0 | 475.6 | 63.56 |
| 14res | 0 | 64.50 | 74.2 | 658.5 | 74.95 |
| 14res | 1 | 63.40 | 73.0 | 656.5 | 74.11 |
| 15res | 0 | 71.80 | 33.0 | 465.8 | 66.73 |
| 15res | 1 | 73.60 | 32.5 | 484.5 | 66.87 |
| 16res | 0 | 72.90 | 46.8 | 490.8 | 75.10 |
| 16res | 1 | 74.00 | 46.1 | 500.0 | 74.81 |

Ba điều phải nói rõ trong báo cáo, nếu không người đọc sẽ tưởng bảng sai:

- **`dev_f1` không so được với `test_f1`.** Hàm `evaluate` trong `T5FineTuner` chấm dev
  bằng giải mã **không ràng buộc, 1 góc nhìn, không bỏ phiếu**, còn test mới chạy đủ
  công thức paper (ràng buộc + 5 góc nhìn + bỏ phiếu). Vì vậy 14res có dev 64.5 mà
  test 74.95, còn 15res ngược lại (dev 71.8, test 66.3 — dev của 15res dễ hơn test).
  Chỉ số dev dùng để vẽ đường hội tụ và chứng minh mô hình có học, **không** dùng để
  chọn checkpoint (MvP giữ epoch cuối).
- **`ms_moi_cau` là cho cả 5 góc nhìn**, không phải một lượt sinh. Chia cho K nếu bảng
  chi phí cần "mỗi góc nhìn". Đo trên T4, chỉ tính phần `generate` của tập test, không
  tính nạp mô hình và tokenize.
- **`mb_gpu` để trống** vì `pytorch-lightning` không in `max_memory_allocated`. Muốn có
  thì phải chạy lại một lần suy luận có gọi hàm đó. Ước lượng từ kích thước mô hình:
  T5-base 892 MB tham số, batch 16 × 200 token, đỉnh thực tế khoảng 3–4 GB trên T4
  15 GB — nhưng đừng ghi ước lượng vào bảng như số thật.

## 8. DMASTE zero-shot

Theo kế hoạch nhóm (mục 5): dùng checkpoint đã train trên ASTE-Data-V2, chạy thẳng lên
bốn domain đích (`book`, `pet`, `toy`, `grocery` — chỉ có dev/test), không train lại.
Bỏ các bộ ba có khía cạnh ẩn vì MvP không được huấn luyện để sinh ra chúng, và kế
hoạch cũng ghi "nếu có thì lọc".

```bash
python models/multi-view-prompting/prepare_dmaste.py --target-only --verify
CKPT=results/runs/mvp/laptop14/seed0/final \
MAX_SEQ_LENGTH=256 EVAL_BATCH_SIZE=32 \
  bash models/multi-view-prompting/run.sh dmaste book
python models/multi-view-prompting/eval_dmaste.py --csv results/MVP_DMASTE.csv
```

Hai tham số phải đổi so với ASTE-Data-V2, nếu không kết quả sẽ sai:

- **`MAX_SEQ_LENGTH=256`.** Câu DMASTE + prompt dài tới 242 token; để mặc định 200
  thì truncation cắt mất span trong khi chỉ số span vẫn tính trên câu gốc.
- **`EVAL_BATCH_SIZE=32`.** 256 token × batch 64 dễ OOM trên T4 15 GB.

Bỏ khía cạnh ẩn làm bài toán dễ đi đáng kể: hơn 40% bộ ba của bốn domain đích là ẩn
(book 404/1068, pet 523/1104, toy 611/1431, grocery 514/1281), nên số zero-shot ở đây
**không so trực tiếp** với bảng in-domain của paper DMASTE.

Số liệu điền vào `results/runs.csv` sau khi chạy notebook, cột `ghi_chu` ghi rõ
"zero-shot, bỏ mẫu khía cạnh ẩn".
