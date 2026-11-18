# Phân công công việc (nhóm 4 người)

> Giả định: 4 người, 6 tuần, huấn luyện trên Kaggle (GPU T4). A, B, C, D là ký hiệu tạm.
> Kế hoạch chi tiết: `ke-hoach-trien-khai-aste.md`.

## 1. Ai làm gì

| | Mô hình tự chạy | Việc khác |
|---|---|---|
| **A** | EMC-GCN | Bảng so sánh, bảng chi phí, DMASTE, slide |
| **B** | Span-ASTE | Phân tích lỗi |
| **C** | MvP | Demo Gradio, thử số góc nhìn của MvP |
| **D** | GTS-BERT, GTS-BiLSTM, baseline luật, baseline CRF | Script đánh giá chung, ghép báo cáo |

Mỗi người tự viết phần báo cáo và tự giải thích lỗi của mô hình mình chạy.

## 2. Việc theo tuần

| Tuần | A | B | C | D |
|---|---|---|---|---|
| 1 | Vá EMC-GCN, chạy thử, đo thời gian | Dựng môi trường Python 3.7, chạy thử Span-ASTE, đo thời gian | Cài thư viện cũ, chạy thử MvP, đo thời gian | Nộp danh sách paper. Viết script đánh giá chung. Vá GTS, chạy thử, đo thời gian |
| 2–3 | Huấn luyện EMC-GCN. Chạy hộ MvP cho C nếu cần | Huấn luyện Span-ASTE | Huấn luyện MvP | Huấn luyện GTS. Viết baseline luật. Gom số baseline từ paper |
| 4 | Lập bảng so sánh và bảng chi phí. Chuẩn bị DMASTE | Viết công cụ phân tích lỗi | Dựng demo | Viết baseline CRF. Chấm điểm toàn bộ, giao số cho A |
| 5 | Chạy EMC-GCN trên DMASTE, lập bảng kết quả | Phân tích lỗi trên 14lap và 14res. Chạy Span-ASTE trên DMASTE | Thử số góc nhìn. Chạy MvP trên DMASTE | Chấm kết quả DMASTE |
| 6 | Viết báo cáo, làm slide | Viết báo cáo | Viết báo cáo, chạy thử demo | Viết báo cáo, ghép báo cáo, dọn repo |

"Huấn luyện" nghĩa là chạy trên 4 tập, mỗi tập 2 seed (`0` và `1`). Cuối tuần 1 mọi người báo thời gian một lần chạy, để nhóm biết tổng giờ GPU có đủ không.

## 3. Việc phải làm ở tuần 1 với từng repo

Mã gốc không chạy được ngay trên Kaggle và không ghi file dự đoán. Mỗi người phải sửa repo của mình:

- **EMC-GCN (A):** sửa lệnh gọi BERT (`code/model.py:151`), import `AdamW`, `torch.load`; đặt tên checkpoint theo tập và seed; thêm đoạn ghi file dự đoán (`code/main.py:161`). Dùng dữ liệu `data/D2`.
- **Span-ASTE (B):** cần Python 3.7, `torch==1.7`, `allennlp==1.2.2` (dùng micromamba). Sửa đường dẫn dữ liệu. Đổi `pred.txt` sang format chung.
- **MvP (C):** cài `transformers==4.14.1`, `pytorch_lightning==1.9.5`. Sửa `scripts/run_main.sh` để chạy đủ 4 tập. Thêm đoạn ghi file dự đoán (`src/main.py:586`).
- **GTS (D):** sửa lệnh gọi BERT (giống EMC-GCN, lấy bản vá của A), `lengths.cpu()`, `torch.load`, bỏ lệnh `input()`; thêm đặt seed. Dùng dữ liệu `models/EMCGCN-ASTE/data/D2`.

Mọi chỗ sửa phải commit vào repo nhóm, không sửa trong notebook.

## 4. Quy ước chung

Xem `quy-uoc-chung.md`: tên gọi, seed, siêu tham số, format file dự đoán, cách chấm điểm, cách đo chi phí, nhật ký `results/runs.csv`, notebook và git.

## 5. Mốc giao nhận

| Hạn | Ai giao | Nội dung |
|---|---|---|
| Cuối tuần 1 | D | Script đánh giá chung và hàm cấp `id` |
| Cuối tuần 1 | Cả nhóm | Thời gian một lần chạy, để tính tổng giờ GPU |
| Cuối tuần 3 | Cả nhóm | Toàn bộ file dự đoán và checkpoint |
| Giữa tuần 4 | D | Bảng điểm đầy đủ, giao cho A |
| Cuối tuần 4 | B | Công cụ phân tích lỗi |
| Cuối tuần 4 | A | Dữ liệu DMASTE cho B và C |
| Giữa tuần 6 | A, B, C | Bản nháp báo cáo, giao cho D |

## 6. Rủi ro cần biết

- **D nặng việc ở tuần 3–4.** Nếu trễ thì bỏ baseline CRF, giữ baseline luật.
- **GTS-BiLSTM mặc định 600 epoch.** Nếu quá lâu thì thêm dừng sớm và ghi lại.
- **Span-ASTE khó dựng môi trường.** Nếu hết tuần 1 chưa được thì thử Colab theo `demo.ipynb` của tác giả.
- **EMC-GCN trên DMASTE** cần tự phân tích cú pháp, kết quả có thể thấp giả. Nếu không tin được thì bỏ EMC-GCN khỏi phần DMASTE.
- **Demo:** tối thiểu là MvP chạy trực tiếp, hai mô hình kia hiển thị dự đoán đã lưu sẵn.

## 7. Việc cần chốt

- Điền tên thật cho A, B, C, D.
- Hạn nộp.
- Repo để công khai hay riêng tư (riêng tư thì Kaggle phải clone bằng token).
- Báo giảng viên danh sách paper và baseline.
