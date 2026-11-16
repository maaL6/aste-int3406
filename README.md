# So sánh các phương pháp ASTE

Bài tập lớn môn Xử lý ngôn ngữ tự nhiên: so sánh các phương pháp trích xuất bộ ba khía cạnh – ý kiến – cảm xúc (Aspect Sentiment Triplet Extraction) trên ASTE-Data-V2. Bối cảnh, kế hoạch và phân công nằm trong `docs/`.

Mọi mô hình được fine-tune từ backbone pretrained (`bert-base-uncased`, `t5-base`) trên cùng dữ liệu và cùng cách chia train/dev/test. Trọng số backbone không nằm trong repo, code tự tải từ Hugging Face khi chạy.

## Cấu trúc

```
docs/       bối cảnh đề bài, kế hoạch triển khai, phân công
data/       ASTE-Data-V2 gốc và script kiểm tra dữ liệu
models/     mã nguồn của tác giả (mỗi mô hình một thư mục)
notebooks/  notebook huấn luyện trên Kaggle
eval/       script đánh giá chung, phân tích lỗi
results/    log, file dự đoán (results/preds/{mô hình}/{tập}/seed{n}.jsonl), runs.csv
demo/       ứng dụng Gradio
```

## Nguồn mã

Mã trong `models/` được copy nguyên trạng từ repo của tác giả trong commit "Vendor upstream model repos (unmodified)". Mọi chỉnh sửa của nhóm nằm ở các commit sau đó; xem bằng `git log -- models/<tên>` hoặc `git diff <commit vendor> -- models/<tên>`.

| Thư mục | Paper | Repo gốc | Commit |
|---|---|---|---|
| `models/EMCGCN-ASTE` | EMC-GCN (ACL 2022) | [CCChenhao997/EMCGCN-ASTE](https://github.com/CCChenhao997/EMCGCN-ASTE) | `4af2ca6` |
| `models/GTS` | GTS (Findings of EMNLP 2020) | [NJUNLP/GTS](https://github.com/NJUNLP/GTS) | `bbb5a6c` |
| `models/Span-ASTE` | Span-ASTE (ACL 2021) | [chiayewken/Span-ASTE](https://github.com/chiayewken/Span-ASTE) | `b252f8b` |
| `models/multi-view-prompting` | MvP (ACL 2023) | [ZubinGou/multi-view-prompting](https://github.com/ZubinGou/multi-view-prompting) | `549a2b7` |

## Dữ liệu

Nguồn gốc: `data/ASTE-Data-V2/`, lấy từ thư mục `ASTE-Data-V2-EMNLP2020` của [xuuuluuu/SemEval-Triplet-data](https://github.com/xuuuluuu/SemEval-Triplet-data) (commit `d0df660`).

Mỗi repo mô hình đã kèm sẵn bản V2 ở định dạng riêng, nên không cần chuyển format. Mỗi mô hình đọc đúng thư mục sau:

| Mô hình | Dữ liệu dùng | Không dùng |
|---|---|---|
| EMC-GCN | `models/EMCGCN-ASTE/data/D2/{lap14,res14,res15,res16}` | `data/D1` (bản V1) |
| GTS | `models/EMCGCN-ASTE/data/D2/...` (bản `data/ASTE_DATA_V2` của GTS thiếu trường `id` mà code cần) | `models/GTS/data/{lap14,...}` (bản V1) |
| Span-ASTE | giải nén `models/Span-ASTE/data.zip` → `data/triplet_data/{14lap,...}` | |
| MvP | `models/multi-view-prompting/data/aste/{laptop14,rest14,rest15,rest16}` | |

Kiểm tra mọi bản đi kèm trùng với bản gốc (cùng câu, cùng bộ ba):

```bash
python data/check_data.py
```

| Tập | Train (câu / bộ ba) | Dev | Test |
|---|---|---|---|
| 14lap | 906 / 1460 | 219 / 346 | 328 / 543 |
| 14res | 1266 / 2338 | 310 / 577 | 492 / 994 |
| 15res | 605 / 1013 | 148 / 249 | 322 / 485 |
| 16res | 857 / 1394 | 210 / 339 | 326 / 514 |
