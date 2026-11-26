# Thí nghiệm số góc nhìn K của MvP

Sinh bởi `models/multi-view-prompting/ablation_table.py`; F1 chấm bằng `eval/aste_eval.py` (so trên văn bản), seed 0, dùng lại checkpoint đã train nên chỉ tốn thời gian suy luận.

| Tập | K=1 | K=2 | K=3 | K=5 |
|---|---:|---:|---:|---:|
| 14lap | 63.30 | 61.58 | 63.36 | 63.00 |
| 14res | 73.83 | 72.99 | 74.50 | 74.75 |
| 15res | 64.89 | 63.56 | 65.93 | 66.33 |
| 16res | 73.62 | 73.65 | 74.41 | 74.90 |
| **Trung bình** | **68.91** | **67.94** | **69.55** | **69.75** |
