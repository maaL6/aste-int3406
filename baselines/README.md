# Baseline không dùng học sâu

Hai baseline do nhóm tự viết, không phải phương pháp dựng lại từ paper nào.

| File | Nội dung |
|---|---|
| `rule.py` | Luật trên cây cú pháp phụ thuộc + từ điển cảm xúc, không huấn luyện |
| `crf.py` | CRF gán nhãn khía cạnh và ý kiến, rồi hai mô hình hồi quy logistic ghép cặp và phân loại cảm xúc |
| `common.py` | Đọc dữ liệu, từ điển, các hàm trên cây cú pháp |
| `lexicon/` | Từ điển cảm xúc của Hu & Liu (KDD 2004), tải từ <https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html>; file từ tiêu cực đã đổi sang UTF-8 |

Chạy từ thư mục gốc của repo, trên CPU, mỗi lệnh dưới một phút:

```bash
python baselines/rule.py            # ghi results/preds/rule/{tập}/seed0.jsonl
python baselines/rule.py --tune     # so các biến thể của bộ luật trên dev

pip install -r baselines/requirements.txt
python baselines/crf.py             # ghi results/preds/crf/{tập}/seed0.jsonl
```

Từ loại và cây cú pháp lấy từ dữ liệu đi kèm EMC-GCN (`models/EMCGCN-ASTE/data/D2`), nên không cần cài bộ phân tích cú pháp. Thời gian suy luận đo được vì vậy không gồm thời gian phân tích cú pháp.

Cả hai baseline không có tính ngẫu nhiên, chỉ có `seed0`.
