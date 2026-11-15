# Kế hoạch triển khai bài tập lớn NLP: So sánh các phương pháp ASTE

> **Giả định của bản kế hoạch:** nhóm 3 người, thời gian 6 tuần, chạy trên Google Colab / Kaggle (GPU T4 hoặc P100). Nếu khác, điều chỉnh lại mục 8 và mục 9.
> Các con số F1 công bố và số câu của từng tập được ghi theo trí nhớ, **cần đối chiếu lại với paper trước khi nộp**. Các đường dẫn paper, repo và dữ liệu đã được tra lại.

## 1. Tổng quan đề tài

**Đề tài:** So sánh các phương pháp trích xuất bộ ba khía cạnh – ý kiến – cảm xúc (Aspect Sentiment Triplet Extraction, ASTE).

**Đầu vào / đầu ra:** nhận một câu đánh giá, trả về tập các bộ ba (khía cạnh, ý kiến, cảm xúc).

Ví dụ minh họa: "Pin bền nhưng màn hình tối" → (pin, bền, POS), (màn hình, tối, NEG). Dữ liệu thực nghiệm là tiếng Anh.

### 1.1. Lý do chọn đề tài

- Phân tích cảm xúc mức câu chỉ cho biết đánh giá tích cực hay tiêu cực. ASTE trả lời đồng thời ba câu hỏi: khen chê khía cạnh nào, bằng ý kiến gì, cảm xúc ra sao. Kết quả này dùng trực tiếp được cho phân tích đánh giá sản phẩm, dịch vụ.
- Đây là tác vụ đầy đủ nhất trong nhóm ABSA dạng trích xuất: mô hình phải vừa xác định biên span, vừa ghép cặp khía cạnh – ý kiến, vừa phân loại cảm xúc.
- Có benchmark chuẩn (ASTE-Data-V2) được hầu hết nghiên cứu sử dụng, nên so sánh công bằng được.
- Bài toán đã phát triển qua nhiều hướng tiếp cận: pipeline, gán nhãn, span, sinh văn bản.
- Dữ liệu nhỏ, phù hợp tài nguyên của nhóm.

### 1.2. Mục tiêu

1. Dựng lại kết quả của ba paper trên ASTE-Data-V2, đối chiếu với kết quả công bố và giải thích chênh lệch.
2. So sánh ba hướng tiếp cận cùng các baseline truyền thống trên cùng dữ liệu, cách chia và độ đo; ghi nhận thời gian huấn luyện, số tham số, phần cứng.
3. Phân tích lỗi theo từng loại và lý giải từ cơ chế của mỗi phương pháp.
4. (Điểm cộng) Đánh giá khả năng tổng quát hóa trên dữ liệu khác miền và xây dựng demo.

## 2. Paper và baseline

### 2.1. Ba paper chính

| Hướng tiếp cận | Paper | Hội nghị | Backbone | Paper | Mã nguồn chính thức |
|---|---|---|---|---|---|
| Gán nhãn lưới | EMC-GCN: Enhanced Multi-Channel Graph Convolutional Network for ASTE | ACL 2022 | BERT-base | [Paper](https://aclanthology.org/2022.acl-long.212) | [GitHub](https://github.com/CCChenhao997/EMCGCN-ASTE) |
| Trích xuất theo span | Span-ASTE: Learning Span-Level Interactions for ASTE | ACL 2021 | BERT-base | [Paper](https://aclanthology.org/2021.acl-long.367.pdf) | [GitHub](https://github.com/chiayewken/Span-ASTE) |
| Sinh văn bản | MvP: Multi-view Prompting Improves Aspect Sentiment Tuple Prediction | ACL 2023 | T5-base | [Paper](https://aclanthology.org/2023.acl-long.240.pdf) | [GitHub](https://github.com/ZubinGou/multi-view-prompting) |

**Tóm tắt cơ chế:**

- **EMC-GCN:** coi câu là một lưới từ – từ, mỗi ô được gán nhãn quan hệ. Mô hình dùng GCN đa kênh, kết hợp thêm đặc trưng ngôn ngữ học (từ loại, cú pháp, khoảng cách) để dự đoán nhãn từng ô, rồi giải mã lưới thành bộ ba.
- **Span-ASTE:** liệt kê mọi span ứng viên, phân loại span nào là khía cạnh hoặc ý kiến, lọc bớt, rồi ghép cặp và phân loại cảm xúc ở mức span. Nhờ biểu diễn mức span, mô hình xử lý tốt các cụm nhiều từ.
- **MvP:** đưa bài toán về dạng sinh văn bản với T5. Mỗi bộ ba được sinh dưới dạng chuỗi có đánh dấu phần tử, ví dụ `[A] pin [O] bền [S] tích cực`. Thứ tự các phần tử (A, O, S) được hoán vị và đưa vào prompt, mỗi thứ tự là một "góc nhìn". Khi huấn luyện, mỗi câu được ghép với nhiều thứ tự, chọn theo điểm entropy trên tập huấn luyện. Khi suy luận, mô hình sinh kết quả theo từng thứ tự rồi bỏ phiếu, chỉ giữ các bộ ba xuất hiện ở đa số góc nhìn.

**Lý do chọn:** cả ba cùng tác vụ, cùng benchmark ASTE-Data-V2, đều đăng ở ACL (CORE A*), có mã nguồn chính thức, và đại diện cho ba hướng tiếp cận nối tiếp nhau trong giai đoạn 2021–2023.

### 2.2. Baseline truyền thống

| Baseline | Nguồn | Hướng | Cách dùng | Liên kết |
|---|---|---|---|---|
| Peng et al. two-stage | AAAI 2020 | Pipeline hai giai đoạn | Trích số từ paper | [Paper](https://arxiv.org/abs/1911.01616) |
| CMLA+, RINANTE+, Li-unified-R+ | Biến thể trong Peng et al. | Pipeline | Trích số từ paper | [Paper](https://arxiv.org/abs/1911.01616) |
| OTE-MTL | Findings of EMNLP 2020 | Học đa nhiệm | Trích số từ paper | [Paper](https://arxiv.org/abs/2010.01512), [GitHub](https://github.com/GeneZC/OTE-MTL) |
| JET | EMNLP 2020 | Gán nhãn theo vị trí | Trích số (chạy lại nếu còn thời gian) | [Paper](https://arxiv.org/abs/2010.02609), [GitHub](https://github.com/xuuuluuu/Position-Aware-Tagging-for-ASTE) |
| GTS-BiLSTM, GTS-BERT | Findings of EMNLP 2020 | Gán nhãn lưới | **Tự chạy lại** | [Paper](https://aclanthology.org/2020.findings-emnlp.234), [GitHub](https://github.com/NJUNLP/GTS) |
| BARTABSA | ACL 2021 | Sinh chỉ số con trỏ | Trích số, làm mốc cho hướng sinh | [Paper](https://arxiv.org/pdf/2106.04300), [GitHub](https://github.com/yhcc/BARTABSA) |

Số của các baseline trên ASTE-Data-V2 lấy từ bảng kết quả trong paper Span-ASTE và EMC-GCN. Trong báo cáo phải đánh dấu rõ số nào tự chạy, số nào trích.

## 3. Dữ liệu

### 3.1. Benchmark chính: ASTE-Data-V2

Nguồn:

- **Bản gốc (nguồn chính):** [xuuuluuu/SemEval-Triplet-data](https://github.com/xuuuluuu/SemEval-Triplet-data), thư mục [ASTE-Data-V2-EMNLP2020](https://github.com/xuuuluuu/SemEval-Triplet-data/tree/master/ASTE-Data-V2-EMNLP2020). Có chỉ số vị trí token.
- **Bản Hugging Face có chỉ số vị trí:** [matthewfranglen/aste-v2](https://huggingface.co/datasets/matthewfranglen/aste-v2), dựng lại từ repo gốc.
- **Bản Hugging Face dạng sinh văn bản:** [NEUDM/aste-data-v2](https://huggingface.co/datasets/NEUDM/aste-data-v2). Đã chuyển sang dạng instruction cho mô hình sinh, không phù hợp làm nguồn cho EMC-GCN và Span-ASTE.

| Tập | Train | Dev | Test |
|---|---|---|---|
| 14res | 1266 | 310 | 492 |
| 14lap | 906 | 219 | 328 |
| 15res | 605 | 148 | 322 |
| 16res | 857 | 210 | 326 |

(Số câu, cần đối chiếu lại khi tải dữ liệu.)

### 3.2. Dữ liệu bổ sung

| Bộ dữ liệu | Mục đích | Ghi chú | Liên kết |
|---|---|---|---|
| DMASTE (Findings of ACL 2023) | Kiểm tra tổng quát hóa khác miền | Ưu tiên số 1 cho phần điểm cộng. Có khía cạnh ẩn, cần lọc | [Paper](https://aclanthology.org/2023.findings-acl.178), [GitHub](https://github.com/NJUNLP/DMASTE) |
| Domain-expanded ASTE (DAMO-NLP-SG) | Kiểm tra tổng quát hóa khác miền | Lựa chọn thay thế cho DMASTE | [GitHub](https://github.com/DAMO-NLP-SG/domain-expanded-aste) |
| ASQP Rest15/Rest16 (EMNLP 2021) | Thêm dữ liệu cùng miền | Bỏ phần danh mục, lọc mẫu ẩn. Cùng nguồn SemEval với 15res, 16res nên phải lọc câu trùng | [Paper](https://aclanthology.org/2021.emnlp-main.726.pdf), [GitHub](https://github.com/IsakZhang/ABSA-QUAD) |
| ACOS Restaurant/Laptop (ACL 2021) | Thêm dữ liệu cùng miền | Bỏ phần danh mục, lọc mẫu có khía cạnh hoặc ý kiến ẩn | [Paper](https://aclanthology.org/2021.acl-long.29.pdf), [GitHub](https://github.com/NUSTM/ACOS) |
| Tập tiếng Việt tự gán nhãn (100–200 câu) | Thử nghiệm ngoài tiếng Anh | Tùy chọn, chỉ làm nếu còn thời gian | Không có |

### 3.3. Quy tắc tiền xử lý

- Lấy bản gốc có chỉ số vị trí làm nguồn duy nhất, chuyển sang format của từng repo. Không chuyển ngược từ văn bản sang chỉ số, vì sẽ mơ hồ khi một từ lặp lại trong câu.
- Giữ nguyên cách chia train/dev/test gốc cho mọi mô hình.
- Viết một script `convert.py` cho mỗi format đích và một script kiểm tra: số câu và số bộ ba sau khi chuyển phải khớp với bản gốc.

## 4. Chiến lược huấn luyện

- **Dùng backbone pretrained, tự fine-tune trên dữ liệu ASTE.** Đây là cách cả ba paper làm.
- **Không** train backbone từ đầu: dữ liệu chỉ khoảng một nghìn câu mỗi tập.
- **Không** chỉ tải checkpoint ASTE có sẵn về suy luận: sẽ không đo được thời gian huấn luyện và không giải thích được chênh lệch.
- GTS-BiLSTM là mô hình duy nhất không dùng mô hình ngôn ngữ pretrained (chỉ dùng word embedding có sẵn), đóng vai trò mốc đối chiếu.
- Siêu tham số: dùng cấu hình mặc định của tác giả cho bước reproduce. Mọi thay đổi (giảm batch size, giảm epoch do giới hạn GPU) phải ghi lại.
- Mỗi cấu hình chạy 3–5 seed, báo cáo trung bình và độ lệch chuẩn.

## 5. Các giai đoạn triển khai

### Giai đoạn 0: Duyệt đề tài

- [ ] Hoàn thiện danh sách paper, lý do chọn, đường dẫn mã nguồn.
- [ ] Kiểm tra lại hạng hội nghị trên CORE, số trích dẫn trên Google Scholar.
- [ ] Nộp giảng viên duyệt.

### Giai đoạn 1: Dựng môi trường và dữ liệu

- [ ] Clone bốn repo (EMC-GCN, Span-ASTE, MvP, GTS), mỗi repo một notebook riêng.
- [ ] Cài đúng phiên bản thư viện theo `requirements` của từng repo. Span-ASTE dùng AllenNLP bản cũ, làm sớm nhất.
- [ ] Tải ASTE-Data-V2, chuyển format, chạy script kiểm tra.
- [ ] Chạy thử mỗi mô hình 1 epoch trên 14lap để chắc chắn pipeline thông.

**Kết quả cần có:** bốn notebook chạy được từ đầu đến cuối, có lưu log.

### Giai đoạn 2: Reproduce

- [ ] Huấn luyện từng mô hình trên 4 tập, 3–5 seed.
- [ ] Ghi lại cho mỗi lần chạy: P, R, F1 trên dev và test, thời gian huấn luyện, số tham số, loại GPU, bộ nhớ GPU cao nhất.
- [ ] Lưu file dự đoán trên tập test của mọi lần chạy (dùng cho giai đoạn 4).
- [ ] So với số công bố, ghi nguyên nhân chênh lệch.

Bảng đối chiếu (F1 trên test, điền cột "Nhóm"):

| Mô hình | 14lap (paper / nhóm) | 14res | 15res | 16res |
|---|---|---|---|---|
| EMC-GCN | ~58.8 / … | ~71.8 / … | ~61.9 / … | ~68.3 / … |
| Span-ASTE | ~59.4 / … | ~71.9 / … | ~63.3 / … | ~70.3 / … |
| MvP | ~63.3 / … | ~74.1 / … | ~65.9 / … | ~73.5 / … |
| GTS-BERT | tra bảng Span-ASTE / … | … | … | … |

Các nguyên nhân chênh lệch thường gặp cần kiểm tra: khác seed, khác phiên bản thư viện, giảm batch size, khác GPU, tập test nhỏ nên dao động khoảng 1 điểm F1.

### Giai đoạn 3: So sánh thống nhất

- [ ] Viết một script đánh giá chung: đưa đầu ra của mọi mô hình về cùng dạng tập bộ ba cho mỗi câu, rồi tính P, R, F1 (đúng khi khớp chính xác cả ba phần tử).
- [ ] Lưu ý: MvP so khớp trên văn bản, hai mô hình còn lại so khớp trên chỉ số. Thống nhất một cách (đề xuất: so trên văn bản đã chuẩn hóa chữ thường) và ghi rõ trong báo cáo.
- [ ] Lập bảng tổng hợp: ba mô hình chính, GTS tự chạy, baseline trích số.
- [ ] Lập bảng chi phí: số tham số, thời gian huấn luyện, thời gian suy luận mỗi câu, phần cứng.
- [ ] Nêu rõ khác biệt backbone (BERT-base khoảng 110M, T5-base khoảng 220M tham số) khi kết luận.

### Giai đoạn 4: Phân tích lỗi

- [ ] Phân loại lỗi trên tập test của 14lap và 14res:
  - Sai biên span (thừa hoặc thiếu từ).
  - Đúng span nhưng ghép cặp sai.
  - Đúng cặp nhưng sai cảm xúc.
  - Bỏ sót bộ ba (đặc biệt khi một khía cạnh có nhiều ý kiến hoặc ngược lại).
  - Dự đoán thừa bộ ba.
  - Sinh sai định dạng hoặc sinh từ không có trong câu (riêng MvP).
- [ ] Thống kê F1 theo: độ dài câu, số bộ ba mỗi câu, span một từ so với nhiều từ, bộ ba chồng lấn.
- [ ] Chọn 10–15 ví dụ tiêu biểu mà các mô hình dự đoán khác nhau.
- [ ] Giải thích từng loại lỗi từ cơ chế mô hình. Gợi ý giả thuyết cần kiểm chứng:
  - Gán nhãn lưới dễ lỗi ở span dài, vì phải đúng mọi ô trong lưới.
  - Span-ASTE dễ lỗi khi bước lọc span loại nhầm ứng viên đúng.
  - MvP dễ lỗi định dạng, nhưng bỏ phiếu giúp tăng precision và có thể giảm recall.

### Giai đoạn 5: Điểm cộng

- [ ] **Tổng quát hóa:** chạy ba mô hình đã train trên ASTE-Data-V2 lên DMASTE mà không train lại, so mức sụt giảm F1.
- [ ] **Demo Gradio:** nhập một câu, hiển thị bộ ba từ ba mô hình cạnh nhau, tô màu span trong câu.
- [ ] **Cải tiến nhỏ (chọn một nếu còn thời gian):**
  - Thay đổi số góc nhìn của MvP và đo đánh đổi giữa F1 và thời gian.
  - Kết hợp đầu ra ba mô hình bằng bỏ phiếu.
  - Thay backbone của Span-ASTE hoặc EMC-GCN bằng mô hình lớn hơn để tách ảnh hưởng của backbone.

### Giai đoạn 6: Báo cáo và thuyết trình

- [ ] Viết báo cáo theo dàn ý ở mục 7.
- [ ] Ghi rõ nguồn mã và phần nhóm tự viết hoặc chỉnh sửa (script chuyển format, script đánh giá chung, phân tích lỗi, demo).
- [ ] Dọn repo của nhóm: README, hướng dẫn chạy lại, log kết quả.
- [ ] Chuẩn bị slide và chạy thử demo.

## 6. Cấu trúc repo của nhóm

```
aste-comparison/
├── data/            # dữ liệu gốc và script chuyển format
├── models/          # mỗi mô hình một thư mục (fork từ repo tác giả)
├── notebooks/       # notebook huấn luyện trên Colab/Kaggle
├── eval/            # script đánh giá chung, phân tích lỗi
├── results/         # log, file dự đoán, bảng kết quả
├── demo/            # ứng dụng Gradio
└── README.md
```

## 7. Dàn ý báo cáo

1. Giới thiệu: bài toán, lý do chọn, mục tiêu.
2. Công trình liên quan: quá trình phát triển từ pipeline đến sinh văn bản.
3. Phương pháp: ba mô hình chính và các baseline.
4. Thiết lập thực nghiệm: dữ liệu, độ đo, siêu tham số, phần cứng.
5. Kết quả reproduce và giải thích chênh lệch.
6. So sánh các phương pháp: chất lượng và chi phí.
7. Phân tích lỗi.
8. Cải tiến và demo.
9. Kết luận, hạn chế.
10. Phụ lục: nguồn mã, phần nhóm tự viết, phân công.

## 8. Tiến độ đề xuất (6 tuần)

| Tuần | Công việc | Kết quả cần có |
|---|---|---|
| 1 | Giai đoạn 0 và 1 | Đề tài được duyệt, bốn repo chạy thử được |
| 2–3 | Giai đoạn 2 | Bảng reproduce đủ seed |
| 4 | Giai đoạn 3 | Bảng so sánh thống nhất, bảng chi phí |
| 5 | Giai đoạn 4 và 5 | Phân tích lỗi, demo, kết quả DMASTE |
| 6 | Giai đoạn 6 | Báo cáo, slide, repo hoàn chỉnh |

## 9. Phân công đề xuất (3 người)

| Thành viên | Giai đoạn 1–2 | Giai đoạn 3–6 |
|---|---|---|
| A | EMC-GCN và GTS | Script đánh giá chung, bảng so sánh |
| B | Span-ASTE | Phân tích lỗi |
| C | MvP | Dữ liệu bổ sung, demo Gradio |

Cả nhóm cùng viết báo cáo; mỗi người viết phần mô hình mình phụ trách.

## 10. Rủi ro và phương án dự phòng

| Rủi ro | Phương án |
|---|---|
| Xung đột thư viện cũ (AllenNLP) | Dùng môi trường riêng với đúng phiên bản Python; nếu không được, dùng bản cài lại công khai và ghi rõ nguồn |
| Colab ngắt phiên giữa chừng | Lưu checkpoint và log lên Drive sau mỗi epoch; ưu tiên Kaggle cho lần chạy dài |
| MvP huấn luyện lâu do nhân dữ liệu theo số góc nhìn | Giảm số seed của MvP trước khi giảm số tập dữ liệu |
| Không đủ thời gian chạy hết | Ưu tiên đủ seed trên 14lap và 14res hơn là 1 seed trên cả 4 tập |
| Kết quả lệch nhiều so với paper | Kiểm tra lại dữ liệu và siêu tham số; nếu vẫn lệch, báo cáo trung thực kèm phân tích nguyên nhân |

## 11. Sản phẩm nộp

- Báo cáo.
- Repo mã nguồn kèm hướng dẫn chạy lại.
- Bảng kết quả và log thực nghiệm.
- Demo Gradio (điểm cộng).
- Slide thuyết trình.
