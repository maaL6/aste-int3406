# Phân công công việc (nhóm 4 người)

Tài liệu này thay cho mục 9 trong `ke-hoach-trien-khai-aste.md`. Nội dung từng giai đoạn vẫn xem ở file kế hoạch; bối cảnh và yêu cầu đề bài xem ở `boi-canh-project.md`.

> **Giả định:** 4 người, 6 tuần, huấn luyện hoàn toàn trên Kaggle. Hạn nộp chưa chốt; khi có hạn thật thì chỉnh lại cột "Tuần".
> A, B, C, D là ký hiệu tạm, nhóm tự điền tên.

## 1. Tổng quan vai trò

| Thành viên | Vai trò chính | Mô hình phụ trách | Vai trò ở nửa sau |
|---|---|---|---|
| A | Hướng gán nhãn lưới | EMC-GCN, GTS-BERT, GTS-BiLSTM | Bảng so sánh tổng hợp, bảng chi phí |
| B | Hướng span | Span-ASTE | Chủ trì phân tích lỗi |
| C | Hướng sinh văn bản | MvP | Demo Gradio, thử nghiệm số góc nhìn |
| D | Dữ liệu và đánh giá | Không huấn luyện mô hình chính | DMASTE, baseline dựa trên luật, tổng hợp báo cáo |

Lý do chia:

- EMC-GCN viết dựa trên mã của GTS và dùng format dữ liệu gần giống, nên một người làm cả hai là hợp lý.
- Span-ASTE dùng AllenNLP bản cũ, môi trường khó dựng nhất, cần một người dành trọn thời gian.
- MvP dùng T5-base và nhân dữ liệu theo số góc nhìn, tốn giờ GPU nhất.
- Dữ liệu và script đánh giá là phần cả ba người kia phụ thuộc, nên cần một người làm xong sớm, ngay từ tuần 1–2.

## 2. Công việc chi tiết từng người

### A: EMC-GCN và GTS

| Tuần | Công việc | Kết quả cần có |
|---|---|---|
| 1 | Clone hai repo, dựng môi trường trên Kaggle, chạy thử 1 epoch trên 14lap | Hai notebook chạy thông, có log |
| 2–3 | Huấn luyện EMC-GCN, GTS-BERT, GTS-BiLSTM trên 4 tập, 3–5 seed | File dự đoán và log của mọi lần chạy |
| 4 | Lập bảng so sánh tổng hợp (ba mô hình chính, GTS, baseline trích số), kiểm định ý nghĩa thống kê giữa các mô hình chính, và bảng chi phí | Hai bảng hoàn chỉnh, có trung bình và độ lệch chuẩn; kết quả kiểm định |
| 5 | Giải thích lỗi của EMC-GCN và GTS; nếu còn thời gian, thử thay backbone lớn hơn | Phần phân tích lỗi của hướng gán nhãn lưới |
| 6 | Viết phần phương pháp và kết quả của EMC-GCN, GTS | Các mục báo cáo tương ứng |

### B: Span-ASTE

| Tuần | Công việc | Kết quả cần có |
|---|---|---|
| 1 | Dựng môi trường AllenNLP trên Kaggle (làm đầu tiên vì rủi ro cao nhất), chạy thử 1 epoch trên 14lap | Notebook chạy thông; ghi lại cách cài để cả nhóm chạy lại được |
| 2–3 | Huấn luyện Span-ASTE trên 4 tập, 3–5 seed | File dự đoán và log của mọi lần chạy |
| 4 | Viết công cụ phân tích lỗi: phân loại lỗi tự động, thống kê F1 theo độ dài câu, số bộ ba, span một từ và nhiều từ, bộ ba chồng lấn | Script trong `eval/`, chạy được trên file dự đoán của mọi mô hình |
| 5 | Chạy phân tích lỗi trên 14lap và 14res, chọn 10–15 ví dụ tiêu biểu, giải thích lỗi của Span-ASTE | Bảng thống kê lỗi và bộ ví dụ |
| 6 | Viết phần Span-ASTE và chương phân tích lỗi | Các mục báo cáo tương ứng |

Nếu đến hết tuần 1 vẫn chưa dựng được môi trường AllenNLP, chuyển sang bản cài lại công khai của Span-ASTE và ghi rõ nguồn.

### C: MvP

| Tuần | Công việc | Kết quả cần có |
|---|---|---|
| 1 | Clone repo, chạy thử 1 epoch trên 14lap, **đo thời gian một lần chạy đầy đủ** | Notebook chạy thông; ước lượng tổng giờ GPU cho MvP |
| 2–3 | Huấn luyện MvP trên 4 tập, 3–5 seed (D chạy hộ seed nếu thiếu quota) | File dự đoán và log của mọi lần chạy |
| 4 | Dựng khung demo Gradio: nhập câu, hiển thị bộ ba từ ba mô hình, tô màu span | Demo chạy được với checkpoint có sẵn |
| 5 | Thử nghiệm thay đổi số góc nhìn của MvP, đo F1 và thời gian suy luận; giải thích lỗi của MvP | Bảng đánh đổi F1 và thời gian; phần phân tích lỗi hướng sinh |
| 6 | Viết phần MvP, phần cải tiến và demo; chuẩn bị chạy thử demo khi thuyết trình | Các mục báo cáo tương ứng |

### D: Dữ liệu và đánh giá

| Tuần | Công việc | Kết quả cần có |
|---|---|---|
| 1 | Tải ASTE-Data-V2 bản gốc, viết `convert.py` cho từng format đích và script kiểm tra số câu, số bộ ba; đưa dữ liệu lên Kaggle Dataset chung của nhóm | Dữ liệu đã chuyển format cho cả bốn repo, script kiểm tra chạy đạt |
| 2 | Viết script đánh giá chung (xem mục 4); gom số baseline từ paper Span-ASTE và EMC-GCN, đánh dấu nguồn | Script trong `eval/`; bảng số baseline có trích nguồn |
| 3 | Chấm mọi file dự đoán đã có bằng script chung; chạy hộ seed cho C nếu cần; đối chiếu số câu và F1 công bố với paper | Bảng reproduce được điền dần; danh sách số liệu đã đối chiếu |
| 4 | Chuẩn bị DMASTE: tải, lọc, chuyển format; viết baseline dựa trên luật nếu giảng viên yêu cầu | Dữ liệu DMASTE sẵn sàng; baseline luật (nếu cần) |
| 5 | Chạy ba mô hình đã huấn luyện trên DMASTE cùng A, B, C; lập bảng sụt giảm F1 | Kết quả khác miền |
| 6 | Ghép báo cáo, thống nhất bảng biểu và văn phong; dọn repo, viết README và hướng dẫn chạy lại | Báo cáo hoàn chỉnh, repo sạch |

## 3. Quy ước dùng Kaggle

- **Một loại GPU cho mọi lần chạy.** Chọn T4 (dùng một GPU, kể cả khi chọn cấu hình T4 x2) và giữ nguyên suốt project, để thời gian huấn luyện và suy luận so sánh được. Ghi loại GPU vào log mỗi lần chạy.
- **Quota.** Mỗi tài khoản có khoảng 30 giờ GPU mỗi tuần (xem số chính xác trong trang cài đặt tài khoản). Cả nhóm có khoảng 120 giờ mỗi tuần. Sau tuần 1, mỗi người báo thời gian một lần chạy đầy đủ để nhóm tính tổng giờ cần và quyết định số seed.
- **Chạy dài bằng "Save Version → Save & Run All".** Notebook chạy nền, không phụ thuộc trình duyệt. Mỗi phiên tối đa khoảng 12 giờ, nên chia mỗi notebook theo từng tập dữ liệu hoặc từng nhóm seed.
- **Lưu kết quả.** Ghi checkpoint, log và file dự đoán vào `/kaggle/working`. Sau mỗi lần chạy, tải file dự đoán và log về repo của nhóm (`results/`). Checkpoint cần dùng lại (demo, DMASTE) thì lưu thành Kaggle Dataset.
- **Dữ liệu dùng chung.** D tạo một Kaggle Dataset chứa dữ liệu đã chuyển format; mọi notebook đọc từ đó, không ai tự chuyển format riêng.
- **Bật Internet trong notebook** để clone repo và tải backbone (cần xác minh số điện thoại cho tài khoản).
- **Phiên bản Python.** Kaggle dùng Python mới, trong khi các repo yêu cầu thư viện cũ. Mỗi notebook ghi rõ lệnh cài và phiên bản đã dùng. Riêng Span-ASTE nhiều khả năng cần tạo môi trường Python cũ ngay trong notebook (ví dụ bằng conda hoặc micromamba).

## 4. Quy ước chung giữa các thành viên

**Format file dự đoán** (mỗi mô hình, mỗi tập, mỗi seed một file JSONL; mỗi dòng một câu):

```json
{"id": "14lap-test-0", "tokens": ["The", "battery", "lasts", "long"],
 "pred": [{"aspect": [1, 1], "opinion": [3, 3], "sentiment": "POS",
           "aspect_text": "battery", "opinion_text": "long"}]}
```

- EMC-GCN, GTS và Span-ASTE ghi đủ cả chỉ số và văn bản. MvP chỉ có văn bản, để trống chỉ số.
- Tên file: `results/preds/{mô hình}/{tập}/seed{n}.jsonl`.
- Script đánh giá chung của D đọc format này và tính P, R, F1 theo hai cách: so trên văn bản chuẩn hóa (cho mọi mô hình) và so trên chỉ số (cho các mô hình có chỉ số).

**Nhật ký thực nghiệm:** một bảng chung (Google Sheet hoặc `results/runs.csv`), mỗi lần chạy một dòng gồm mô hình, tập, seed, siêu tham số đã đổi so với mặc định, P, R, F1 trên dev và test, thời gian huấn luyện, số tham số, bộ nhớ GPU cao nhất, đường dẫn notebook.

**Phân tích lỗi:** B viết công cụ và chủ trì, nhưng mỗi người tự giải thích lỗi của mô hình mình phụ trách.

**Báo cáo:** mỗi người viết phần mô hình của mình; D ghép và thống nhất.

## 5. Các mốc phụ thuộc

| Hạn | Người giao | Người nhận | Nội dung |
|---|---|---|---|
| Giữa tuần 1 | D | A, B, C | Dữ liệu đã chuyển format trên Kaggle Dataset |
| Cuối tuần 1 | A, B, C | Cả nhóm | Thời gian một lần chạy đầy đủ, để chốt số seed |
| Giữa tuần 2 | D | A, B, C | Format file dự đoán và script đánh giá chung |
| Cuối tuần 3 | A, B, C | B, D | Toàn bộ file dự đoán trên tập test |
| Cuối tuần 3 | A, B, C | C, D | Checkpoint tốt nhất của mỗi mô hình (cho demo và DMASTE) |
| Cuối tuần 4 | B | A, C | Công cụ phân tích lỗi |
| Giữa tuần 6 | A, B, C | D | Bản nháp các phần báo cáo |

## 6. Việc cần chốt

- Điền tên thật cho A, B, C, D.
- Hạn nộp, để chỉnh lại cột "Tuần".
- Giảng viên hiểu "phương pháp truyền thống" theo nghĩa nào; nếu gồm phương pháp không dùng học sâu thì D làm baseline dựa trên luật ở tuần 4.
- Nhóm đồng ý thay BARTABSA bằng EMC-GCN và nộp danh sách paper để duyệt trước khi cài đặt.
