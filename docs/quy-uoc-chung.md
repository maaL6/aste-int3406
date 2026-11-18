# Quy ước chung

Mọi thành viên làm theo các quy ước này để kết quả ghép lại được với nhau. Nếu cần làm khác, báo nhóm trước và ghi lại vào cột `ghi_chu` của `results/runs.csv`.

## 1. Tên gọi

Dùng đúng các tên sau trong tên file, tên thư mục, bảng và log.

| Loại | Tên được dùng |
|---|---|
| Mô hình | `emcgcn`, `span-aste`, `mvp`, `gts-bert`, `gts-bilstm`, `rule`, `crf` |
| Tập dữ liệu | `14lap`, `14res`, `15res`, `16res` |
| Phần chia | `train`, `dev`, `test` |
| Cảm xúc | `POS`, `NEG`, `NEU` |

Mỗi repo gọi tập dữ liệu một kiểu (`lap14`, `laptop14`, ...). Chỉ dùng tên riêng đó khi gọi lệnh của repo, còn mọi thứ nhóm ghi ra đều dùng tên trong bảng trên.

## 2. Huấn luyện

- **Seed:** mỗi mô hình chạy 2 seed trên mỗi tập, dùng `0` và `1`.
- **Siêu tham số:** giữ mặc định theo lệnh chạy chính thức của tác giả (bảng dưới). Không tự dò siêu tham số.
- **Chọn checkpoint:** theo cách của tác giả. Không bao giờ dùng tập test để chọn checkpoint hay chọn epoch.
- **Dữ liệu:** dùng đúng thư mục ghi trong `README.md`, giữ nguyên cách chia train/dev/test.
- **Phần cứng:** một GPU T4 trên Kaggle cho mọi lần chạy.
- **Thay đổi bắt buộc** (giảm batch size vì thiếu bộ nhớ, thêm dừng sớm, ...): áp dụng cho mọi tập và mọi seed của mô hình đó, và ghi vào `ghi_chu`.

| Mô hình | Backbone | Epoch | Batch | Learning rate | Chọn checkpoint |
|---|---|---|---|---|---|
| `emcgcn` | `bert-base-uncased` | 100 | 16 | 2e-5 (BERT), 1e-3 (phần còn lại) | F1 dev cao nhất |
| `span-aste` | `bert-base-uncased` | 10 | 1 | 5e-5 (BERT), 1e-3 (phần còn lại) | F1 dev cao nhất |
| `mvp` | `t5-base` | 20 | 16 | 1e-4 | Epoch cuối; 5 góc nhìn, bỏ phiếu |
| `gts-bert` | `bert-base-uncased` | 100 | 32 | 5e-5 | F1 dev cao nhất |
| `gts-bilstm` | Không (GloVe + embedding theo miền) | 600 | 32 | 1e-3 | F1 dev cao nhất |

Người phụ trách kiểm tra lại các số này với mã nguồn trước khi chạy đủ 2 seed.

## 3. Nơi lưu kết quả

| Nội dung | Đường dẫn |
|---|---|
| File dự đoán | `results/preds/{mô hình}/{tập}/seed{n}.jsonl` |
| Log huấn luyện | `results/logs/{mô hình}/{tập}/seed{n}.log` |
| Số đo chi phí của lần chạy | `results/logs/{mô hình}/{tập}/seed{n}.stats.json` |
| Log và số đo khi chạy lại checkpoint trên T4 | `results/logs/{mô hình}/{tập}/seed{n}.t4.log`, `seed{n}.t4.stats.json` |
| Nhật ký các lần chạy | `results/runs.csv` |
| Số trích từ paper | `results/so-trich-tu-paper.md` |
| Notebook | `notebooks/{mô hình}.ipynb` |
| Checkpoint | Kaggle Dataset `aste-ckpt-{mô hình}`, file `{tập}-seed{n}`. Không đưa vào git |

Ví dụ: `results/preds/emcgcn/14lap/seed0.jsonl`. Baseline `rule` và `crf` không có tính ngẫu nhiên, chỉ có `seed0`.

## 4. File dự đoán

Mỗi file là dự đoán trên tập **test** của một mô hình, một tập, một seed. Mỗi dòng là một câu, đủ mọi câu của tập test, kể cả câu không dự đoán ra bộ ba nào (`"pred": []`).

```json
{"id": "14lap-test-0", "tokens": ["The", "battery", "lasts", "long"],
 "pred": [{"aspect": [1, 1], "opinion": [3, 3], "sentiment": "POS",
           "aspect_text": "battery", "opinion_text": "long"}]}
```

- `id`: `{tập}-test-{số dòng}`, số dòng đếm từ 0 trong `data/ASTE-Data-V2/{tập}/test_triplets.txt`. Lấy bằng hàm tra `id` của D, không tự đánh số.
- `tokens`: câu tách theo dấu cách, giữ nguyên chữ hoa chữ thường như bản gốc.
- `aspect`, `opinion`: `[từ đầu, từ cuối]`, đếm theo từ, bắt đầu từ 0, gồm cả hai đầu. Span một từ ghi `[1, 1]`.
- `aspect_text`, `opinion_text`: các từ của span nối bằng dấu cách.
- `mvp` không có chỉ số: ghi `"aspect": null, "opinion": null`, chỉ điền phần văn bản.
- Không ghi trùng một bộ ba hai lần trong cùng một câu.

Trước khi commit, chạy script đánh giá chung trên file của mình. Script báo lỗi nếu thiếu câu hoặc sai format.

```bash
python eval/aste_eval.py results/preds/emcgcn/14lap/seed0.jsonl   # chấm một file
python eval/aste_eval.py --all                                    # chấm mọi file, in bảng trung bình
```

Cách dễ nhất để ghi đúng format là gọi hàm có sẵn trong `eval/aste_eval.py` (chỉ dùng thư viện chuẩn, chạy được từ Python 3.7):

```python
import sys; sys.path.insert(0, "eval")          # đường dẫn tới thư mục eval/ của repo
from aste_eval import write_preds, IdLookup

# preds[i]: các bộ ba của câu ở dòng i của tập test, mỗi bộ ba là
# (khía cạnh đầu, khía cạnh cuối, ý kiến đầu, ý kiến cuối, "POS" | "NEG" | "NEU")
write_preds("results/preds/emcgcn/14lap/seed0.jsonl", "14lap", preds)

# mvp: bộ ba là (văn bản khía cạnh, văn bản ý kiến, cảm xúc)
write_preds("results/preds/mvp/14lap/seed0.jsonl", "14lap", preds, text_only=True)

# Nếu mô hình xáo thứ tự câu: tra id theo nội dung câu, rồi lấy số dòng ở cuối id
lookup = IdLookup("14lap")
lookup.get("Boot time is super fast , around anywhere from 35 seconds to 1 minute .")  # "14lap-test-0"
```

So trên văn bản: đưa về chữ thường và gộp khoảng trắng rồi so; trong một câu, các bộ ba có cùng văn bản tính là một. Bộ ba bị ghi lặp trong file gốc chỉ tính một lần (14lap test có 541 bộ ba khác nhau, không phải 543).

## 5. Chấm điểm và báo cáo số

- **Mọi số P, R, F1 trong báo cáo lấy từ script đánh giá chung** (`eval/`), không lấy số repo in ra. Số của repo chỉ dùng để kiểm tra chéo: nếu lệch quá 0,5 điểm F1 so với script chung thì báo D.
- Một bộ ba đúng khi khớp cả khía cạnh, ý kiến và cảm xúc. P, R, F1 tính gộp trên cả tập test (micro).
- Số ghi theo phần trăm, hai chữ số thập phân: `59.38`, không ghi `0.5938`.
- Kết quả ghi là trung bình của 2 seed, kèm một nửa khoảng cách giữa hai seed, ví dụ `59.38 ± 0.42` nghĩa là hai seed cho 58.96 và 59.80. Tính F1 từng seed trước rồi mới lấy trung bình. Không gọi con số sau dấu ± là độ lệch chuẩn, vì chỉ có 2 lần chạy.
- Chỉ kết luận mô hình này tốt hơn mô hình kia khi cả hai seed của nó đều cao hơn cả hai seed của mô hình kia. Nếu không thì ghi là chưa phân biệt được.
- Bảng nào cũng ghi rõ số seed, và đánh dấu số nào nhóm tự chạy, số nào trích từ paper.

## 6. Đo chi phí

Đo cùng một cách để bảng chi phí so sánh được.

| Đại lượng | Cách đo | Đơn vị |
|---|---|---|
| Thời gian huấn luyện | Từ lúc bắt đầu epoch đầu đến lúc xong epoch cuối, tính cả phần chấm dev sau mỗi epoch. Không tính thời gian cài thư viện và tải backbone | Phút |
| Thời gian suy luận | Thời gian dự đoán cả tập test chia cho số câu, dùng batch size mặc định của repo. Không tính thời gian nạp mô hình | Mili giây mỗi câu |
| Số tham số | `sum(p.numel() for p in model.parameters())`, tính cả backbone | Triệu |
| Bộ nhớ GPU | `torch.cuda.max_memory_allocated()` lúc kết thúc huấn luyện | MB |

## 7. Nhật ký `results/runs.csv`

Mỗi lần chạy xong thêm một dòng. Không sửa dòng của người khác. Lần chạy hỏng hoặc chạy lại thì thêm dòng mới và ghi lý do vào `ghi_chu`, không xóa dòng cũ.

| Cột | Ý nghĩa |
|---|---|
| `mo_hinh`, `tap`, `seed` | Theo tên ở mục 1 |
| `nguoi_chay`, `ngay` | Ai chạy, ngày chạy (`YYYY-MM-DD`) |
| `commit` | Mã commit của repo nhóm lúc chạy |
| `dev_f1` | F1 trên dev do repo in ra |
| `test_p`, `test_r`, `test_f1` | Từ script đánh giá chung, so trên văn bản |
| `test_f1_chi_so` | Từ script đánh giá chung, so trên chỉ số (để trống với `mvp`) |
| `test_f1_repo` | F1 trên test do repo in ra |
| `phut_train`, `ms_moi_cau`, `trieu_tham_so`, `mb_gpu` | Theo mục 6 |
| `ghi_chu` | Mọi thay đổi so với mặc định; để trống nếu không có |

## 8. Notebook

Mỗi notebook có các bước theo đúng thứ tự:

1. Clone repo nhóm và in mã commit (`git rev-parse --short HEAD`).
2. Cài thư viện.
3. In phiên bản Python, torch, transformers và tên GPU (`nvidia-smi`).
4. Huấn luyện. Tập và seed khai báo ở một ô đầu notebook, không sửa rải rác.
5. Ghi file dự đoán, log và các số đo chi phí vào `/kaggle/working`.

Notebook chỉ chứa lệnh cài và lệnh chạy. Muốn sửa mã mô hình thì sửa trong `models/` rồi commit.

## 9. Git

- Mã gốc của tác giả nằm ở commit `fd5f5cf`. Mỗi lần sửa mã trong `models/` là một commit riêng, message bắt đầu bằng tên mô hình, ví dụ `emcgcn: sửa lệnh gọi BERT cho transformers 4`.
- `git pull` trước khi làm, commit nhỏ, đẩy lên thường xuyên. Mỗi người chủ yếu sửa trong thư mục mô hình của mình nên ít khi đụng nhau.
- Không commit checkpoint, không commit output của notebook quá lớn.
- File dự đoán và log được commit vào `results/`.

## 10. Phân tích lỗi và viết báo cáo

**Loại lỗi** (dùng đúng các tên này, công cụ của B gán tự động):

| Tên | Ý nghĩa |
|---|---|
| Sai biên | Đúng vị trí nhưng span thừa hoặc thiếu từ |
| Sai cặp | Khía cạnh và ý kiến đều đúng nhưng ghép sai với nhau |
| Sai cảm xúc | Đúng cặp, sai nhãn cảm xúc |
| Bỏ sót | Bộ ba có trong đáp án nhưng không được dự đoán |
| Dự đoán thừa | Bộ ba được dự đoán nhưng không có trong đáp án |
| Sai định dạng | Đầu ra không đọc được thành bộ ba (chỉ có ở `mvp`) |

**Thuật ngữ:** khía cạnh (aspect), ý kiến (opinion), cảm xúc (sentiment), bộ ba (triplet), span giữ nguyên. Ghi thuật ngữ tiếng Anh trong ngoặc ở lần đầu xuất hiện.

**Phần viết về mỗi mô hình** có cùng bốn mục: ý tưởng chính; thiết lập chạy và những chỗ nhóm đã sửa; kết quả so với paper và lý do chênh lệch; lỗi điển hình và giải thích.
