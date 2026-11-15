# Bối cảnh project: Bài tập lớn môn Xử lý ngôn ngữ tự nhiên

Tài liệu này ghi lại yêu cầu của bài tập lớn và đề tài nhóm đã chọn. Kế hoạch thực hiện chi tiết nằm trong `ke-hoach-trien-khai-aste.md`.

## 1. Tóm tắt

- **Môn học:** Xử lý ngôn ngữ tự nhiên (NLP).
- **Giảng viên:** Vương Yến (theo ghi chú của nhóm).
- **Dạng bài:** bài tập lớn theo nhóm.
- **Việc cần làm:** chọn một bài toán NLP, chọn 2–3 paper SOTA có mã nguồn, dựng lại kết quả, so sánh các phương pháp trên cùng điều kiện, và phân tích lỗi.

## 2. Yêu cầu của đề bài

### 2.1. Gợi ý bài toán

Danh sách chỉ là gợi ý, không bắt buộc. Nhóm có thể tự đề xuất bài toán khác, miễn là đủ lớn để so sánh 2–3 phương pháp, có dữ liệu công khai và được giảng viên duyệt.

| Nhóm bài toán | Ví dụ |
|---|---|
| Phân loại | Phân loại chủ đề văn bản, phân tích cảm xúc, phát hiện ngôn từ độc hại, suy luận ngôn ngữ tự nhiên (NLI), độ tương đồng ngữ nghĩa |
| Gán nhãn chuỗi | Tách từ, gán nhãn từ loại (POS), nhận dạng thực thể có tên (NER) |
| Trích xuất thông tin | Trích xuất quan hệ, phân tích cảm xúc theo khía cạnh (ABSA) |
| Truy hồi | Truy hồi thông tin, tìm kiếm ngữ nghĩa, hỏi đáp (đọc hiểu, miền mở, RAG) |
| Sinh văn bản | Tóm tắt văn bản, dịch máy, sửa lỗi chính tả và ngữ pháp, hội thoại |

### 2.2. Yêu cầu thực hiện

1. **Chọn bài toán.** Đăng ký một bài toán, lấy từ gợi ý hoặc tự đề xuất.
2. **Chọn 2–3 paper SOTA** trong bài toán đó, có mã nguồn, đại diện cho các thế hệ phương pháp khác nhau, công bố sau 2020.
3. **Nộp danh sách paper** để giảng viên duyệt trước khi bắt đầu cài đặt, kèm lý do chọn và đường dẫn mã nguồn.
4. **Dựng lại (reproduce).** Chạy mã nguồn trên bộ dữ liệu mà paper đã dùng (hoặc một phần nếu tài nguyên hạn chế). Ghi lại kết quả đạt được so với kết quả công bố và giải thích chênh lệch.
5. **So sánh các phương pháp** trên cùng một bộ dữ liệu (cả các phương pháp truyền thống nhắc đến trong paper), cùng cách chia train/dev/test và cùng độ đo. Ghi thêm thời gian huấn luyện, số tham số và phần cứng sử dụng.
6. **Phân tích lỗi** và giải thích vì sao từng phương pháp mắc lỗi đó.
7. **(Điểm cộng)** Đề xuất một cải tiến nhỏ, hoặc làm demo (Streamlit, Gradio).

Được dùng mã nguồn của tác giả hoặc bản cài lại công khai, nhưng phải ghi rõ nguồn và chỉ ra phần nhóm tự viết hoặc chỉnh sửa.

### 2.3. Tiêu chí chọn paper

Paper được chấp nhận khi đáp ứng đủ các điều kiện sau:

- **Đăng tại hội nghị hoặc tạp chí uy tín:**

  | Loại | Danh sách |
  |---|---|
  | Hội nghị NLP | ACL, EMNLP, NAACL, EACL, COLING, AACL-IJCNLP, Findings of ACL/EMNLP, CoNLL |
  | Hội nghị AI/ML | NeurIPS, ICML, ICLR, AAAI, IJCAI |
  | Tạp chí | TACL, Computational Linguistics, IEEE/ACM TASLP, JAIR, Natural Language Engineering |

- **Có mã nguồn công khai**, ưu tiên mã chính thức được dẫn trong paper (GitHub, Hugging Face).
- **Dùng bộ dữ liệu công khai**, tải được.
- **Chạy được với tài nguyên của nhóm** (Google Colab, Kaggle, ...), có thể thu nhỏ dữ liệu hoặc mô hình.
- **Ưu tiên** paper có nhiều trích dẫn hoặc là mốc quan trọng của bài toán.

Cách kiểm tra mức uy tín:

- Hội nghị: tra hạng trên CORE Ranking, ưu tiên hạng A*, A.
- Tạp chí: tra Scimago (SJR), ưu tiên Q1, Q2.
- Tham khảo thêm: Google Scholar Metrics.

## 3. Đề tài của nhóm

**Đề tài:** So sánh các phương pháp trích xuất bộ ba "khía cạnh – ý kiến – cảm xúc" (Aspect Sentiment Triplet Extraction, ASTE). Bài toán thuộc nhóm trích xuất thông tin, nhánh ABSA.

**Mục tiêu:** nhận một câu đánh giá và trả về các bộ ba (khía cạnh, ý kiến, cảm xúc).

Ví dụ minh họa: "Pin bền nhưng màn hình tối" → (pin, bền, POS), (màn hình, tối, NEG).

### 3.1. Lựa chọn ban đầu của nhóm

| Tên paper | Năm | Hội nghị | Đường dẫn | Mã nguồn |
|---|---|---|---|---|
| Learning Span-Level Interactions for Aspect Sentiment Triplet Extraction (Span-ASTE) | 2021 | ACL | [Paper](https://aclanthology.org/2021.acl-long.367.pdf) | [GitHub](https://github.com/chiayewken/Span-ASTE) |
| A Unified Generative Framework for Aspect-Based Sentiment Analysis (BARTABSA) | 2021 | ACL | [Paper](https://arxiv.org/pdf/2106.04300) | [GitHub](https://github.com/yhcc/BARTABSA) |
| MVP: Multi-view Prompting Improves Aspect Sentiment Tuple Prediction (MvP) | 2023 | ACL | [Paper](https://aclanthology.org/2023.acl-long.240.pdf) | [GitHub](https://github.com/ZubinGou/multi-view-prompting) |

**Dataset ban đầu:** [NEUDM/aste-data-v2](https://huggingface.co/datasets/NEUDM/aste-data-v2)

**Lý do chọn ban đầu:** ba paper cùng tác vụ, cùng benchmark ASTE-Data-V2, và tiếp cận theo ba cách khác nhau (span, sinh chuỗi, sinh với nhiều góc nhìn).

### 3.2. Điều chỉnh sau khi rà soát

Các điều chỉnh dưới đây là đề xuất, cần nhóm chốt và giảng viên duyệt:

| Hạng mục | Ban đầu | Điều chỉnh | Lý do |
|---|---|---|---|
| Paper thứ hai | BARTABSA | [EMC-GCN](https://aclanthology.org/2022.acl-long.212) (ACL 2022) | BARTABSA và MvP cùng là hướng sinh. Thay bằng một paper gán nhãn lưới để có ba hướng khác nhau: gán nhãn, span, sinh văn bản |
| Baseline truyền thống | Chưa có | Tự chạy GTS; trích số Peng et al. two-stage, CMLA+, RINANTE+, Li-unified-R+, OTE-MTL, JET | Đề yêu cầu so sánh cả phương pháp truyền thống nhắc đến trong paper |
| Nguồn dữ liệu | Bản NEUDM trên Hugging Face | Bản gốc [xuuuluuu/SemEval-Triplet-data](https://github.com/xuuuluuu/SemEval-Triplet-data) | Bản NEUDM đã chuyển sang dạng sinh văn bản, không còn chỉ số vị trí mà hai mô hình trích xuất cần |
| Dữ liệu bổ sung | Chưa có | [DMASTE](https://github.com/NJUNLP/DMASTE) để kiểm tra khác miền | Phần mở rộng cho điểm cộng |

BARTABSA được giữ lại làm mốc tham chiếu cho hướng sinh (trích số từ paper).

## 4. Đối chiếu yêu cầu với đề tài

| Yêu cầu của đề | Trạng thái |
|---|---|
| Bài toán được duyệt | ASTE thuộc ABSA trong danh sách gợi ý; cần giảng viên duyệt |
| 2–3 paper, sau 2020, có mã nguồn | Đạt: ACL 2021, ACL 2022, ACL 2023, đều có mã chính thức |
| Hội nghị uy tín | Đạt: cả ba ở ACL (CORE A*) |
| Đại diện các thế hệ phương pháp | Đạt sau điều chỉnh: gán nhãn lưới, span, sinh văn bản |
| Dữ liệu công khai, cùng benchmark | Đạt: ASTE-Data-V2, cả ba paper đều báo cáo trên bộ này |
| Chạy được với tài nguyên nhóm | Dự kiến đạt: mô hình cỡ base, mỗi tập khoảng một nghìn câu train |
| So sánh cả phương pháp truyền thống | Đã lên danh sách baseline; cần hỏi giảng viên "truyền thống" hiểu theo nghĩa nào |
| Phân tích lỗi | Chưa làm, đã có kế hoạch |
| Điểm cộng | Dự kiến: kiểm tra khác miền và demo Gradio |

## 5. Việc cần chốt

- Nhóm có đồng ý thay BARTABSA bằng EMC-GCN không.
- Số thành viên và hạn nộp, để chỉnh tiến độ và phân công.
- Hỏi giảng viên: "phương pháp truyền thống" có cần gồm phương pháp không dùng học sâu (ví dụ dựa trên luật) không.
- Nộp danh sách paper để duyệt trước khi cài đặt.
