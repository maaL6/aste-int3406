# Số trích từ paper (ASTE-Data-V2, tập test)

Các số dưới đây chép từ bảng trong paper, nhóm không tự chạy. Mỗi ô là P / R / F1 (%), bộ ba đúng khi khớp cả khía cạnh, ý kiến và cảm xúc. Mọi số đã được đối chiếu với bản PDF trên ACL Anthology ngày 2026-11-17.

Khi đưa vào bảng so sánh, đánh dấu các hàng này là "trích từ paper" (xem `docs/quy-uoc-chung.md`, mục 5).

## 1. Baseline đời trước

| Phương pháp | 14lap | 14res | 15res | 16res | Nguồn |
|---|---|---|---|---|---|
| CMLA+ | 30.09 / 36.92 / 33.16 | 39.18 / 47.13 / 42.79 | 34.56 / 39.84 / 37.01 | 41.34 / 42.10 / 41.72 | [1] Bảng 2 |
| RINANTE+ | 21.71 / 18.66 / 20.07 | 31.42 / 39.38 / 34.95 | 29.88 / 30.06 / 29.97 | 25.68 / 22.30 / 23.87 | [1] Bảng 2 |
| Li-unified-R | 40.56 / 44.28 / 42.34 | 41.04 / 67.35 / 51.00 | 44.72 / 51.39 / 47.82 | 37.33 / 54.51 / 44.31 | [1] Bảng 2 |
| Peng et al. two-stage | 37.38 / 50.38 / 42.87 | 43.24 / 63.66 / 51.46 | 48.07 / 57.51 / 52.32 | 46.96 / 64.24 / 54.21 | [1] Bảng 2 |
| OTE-MTL | 49.62 / 41.07 / 44.78 | 62.70 / 57.10 / 59.71 | 55.63 / 42.51 / 47.94 | 60.95 / 53.35 / 56.82 | [2] Bảng 2 |
| JET^t (M=6), BiLSTM | 52.00 / 35.91 / 42.48 | 66.76 / 49.09 / 56.58 | 59.77 / 42.27 / 49.52 | 63.59 / 50.97 / 56.59 | [1] Bảng 2 |
| JET^o (M=6), BiLSTM | 53.03 / 33.89 / 41.35 | 61.50 / 55.13 / 58.14 | 64.37 / 44.33 / 52.50 | 70.94 / 57.00 / 63.21 | [1] Bảng 2 |
| JET^t (M=6), BERT | 53.53 / 43.28 / 47.86 | 63.44 / 54.12 / 58.41 | 68.20 / 42.89 / 52.66 | 65.28 / 51.95 / 57.85 | [1] Bảng 2 |
| JET^o (M=6), BERT | 55.39 / 47.33 / 51.04 | 70.56 / 55.94 / 62.40 | 64.45 / 51.96 / 57.53 | 70.42 / 58.37 / 63.83 | [1] Bảng 2 |
| BARTABSA (BART-base) | 61.41 / 56.19 / 58.69 | 65.52 / 64.99 / 65.25 | 59.14 / 59.38 / 59.26 | 66.60 / 68.68 / 67.62 | [3] Bảng 6 |

## 2. Số công bố của các mô hình nhóm tự chạy

Dùng để so với kết quả reproduce của nhóm.

| Phương pháp | 14lap | 14res | 15res | 16res | Nguồn |
|---|---|---|---|---|---|
| GTS-BiLSTM | 53.35 / 40.99 / 46.31 | 66.13 / 57.91 / 61.73 | 60.10 / 46.89 / 52.66 | 63.28 / 58.56 / 60.79 | [2] Bảng 2 |
| GTS-BERT (bản A) | 57.82 / 51.32 / 54.36 | 67.76 / 67.29 / 67.50 | 62.59 / 57.94 / 60.15 | 66.08 / 69.91 / 67.93 | [2] Bảng 2 |
| GTS-BERT (bản B) | 59.40 / 51.94 / 55.42 | 68.09 / 69.54 / 68.81 | 59.28 / 57.93 / 58.60 | 68.32 / 66.86 / 67.58 | [4] Bảng 4 |
| Span-ASTE (BERT) | 63.44 / 55.84 / 59.38 | 72.89 / 70.89 / 71.85 | 62.18 / 64.45 / 63.27 | 69.45 / 71.17 / 70.26 | [2] Bảng 2 |
| EMC-GCN | 61.70 / 56.26 / 58.81 | 71.21 / 72.39 / 71.78 | 61.54 / 62.47 / 61.93 | 65.62 / 71.30 / 68.33 | [4] Bảng 4 |
| MvP (T5-base) | – / – / 63.33 | – / – / 74.05 | – / – / 65.89 | – / – / 73.48 | [5] Bảng 2 |

## 3. Nguồn

1. Xu et al., "Position-Aware Tagging for Aspect Sentiment Triplet Extraction" (JET), EMNLP 2020. <https://aclanthology.org/2020.emnlp-main.183>
2. Xu et al., "Learning Span-Level Interactions for Aspect Sentiment Triplet Extraction" (Span-ASTE), ACL 2021. <https://aclanthology.org/2021.acl-long.367>
3. Yan et al., "A Unified Generative Framework for Aspect-Based Sentiment Analysis" (BARTABSA), ACL 2021. <https://aclanthology.org/2021.acl-long.188>
4. Chen et al., "Enhanced Multi-Channel Graph Convolutional Network for Aspect Sentiment Triplet Extraction" (EMC-GCN), ACL 2022. <https://aclanthology.org/2022.acl-long.212>
5. Gou et al., "MvP: Multi-view Prompting Improves Aspect Sentiment Tuple Prediction", ACL 2023. <https://aclanthology.org/2023.acl-long.240>

## 4. Lưu ý khi trích

- **CMLA+, RINANTE+, Li-unified-R, Peng two-stage:** không phải số trong paper gốc của các phương pháp này. Đây là số do nhóm tác giả JET chạy lại trên V2 và báo cáo trong [1]. Paper gốc của Peng et al. dùng bản dữ liệu V1.
- **OTE-MTL:** số trong [2] do tác giả Span-ASTE chạy lại trên V2. EMC-GCN [4] cũng chạy lại và ra số khác (14lap 43.42, 14res 58.71, 15res 47.13, 16res 56.96). Bảng trên dùng số của [2].
- **GTS:** paper gốc của GTS dùng bộ dữ liệu riêng, không phải V2, nên không so được. Số trên V2 có hai bản do hai nhóm khác chạy lại: bản A của Span-ASTE [2], bản B của EMC-GCN [4].
- **JET:** paper báo cáo nhiều cấu hình. Các paper sau thường trích hàng JET^o (M=6) + BERT với tên "JET-BERT".
- **Số lần chạy:** Span-ASTE, EMC-GCN và MvP ghi rõ là trung bình 5 seed. JET (và các baseline nó chạy lại) và BARTABSA không ghi số lần chạy.
- **MvP:** paper chỉ báo cáo F1. Hàng trên là mô hình huấn luyện riêng cho từng tập; paper còn có bản huấn luyện gộp nhiều tác vụ (14lap 65.30, 14res 76.30, 15res 69.44, 16res 73.10), không so trực tiếp được với cách nhóm chạy.
- **BARTABSA:** ô P của 16res in trong paper là "66.6".
- **14lap test:** file gốc có 2 bộ ba bị ghi lặp ở một câu. Script của nhóm chỉ tính một lần (541 bộ ba thay vì 543), nên recall của nhóm có thể lệch rất nhẹ so với cách tính của từng paper.
