# Phân tích lỗi MvP (C)

Sinh bởi `models/multi-view-prompting/error_report.py` từ file dự đoán trong `results/preds/mvp/`. Nhãn loại lỗi theo `docs/quy-uoc-chung.md` mục 10. Số P/R/F1 ở đây khớp với `eval/aste_eval.py` vì cùng đọc một file.

## 14lap seed0

- Bộ ba vàng: **541**, dự đoán: **513**, đúng: **332**
- **P = 64.72  R = 61.37  F1 = 63.00** (khớp tuyệt đối bộ ba)

| Loại lỗi | Số lượng | % lỗi phía đáp án |
|---|---:|---:|
| Sai biên | 85 | 40.7% |
| Sai cặp | 12 | 5.7% |
| Sai cảm xúc | 35 | 16.7% |
| Bỏ sót | 77 | 36.8% |
| Dự đoán thừa | 49 | — |
| Sai định dạng | 0 | — |

### Theo độ dài câu

| Độ dài (từ) | Câu | Gold | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| 11–20 | 171 | 294 | 65.82 | 61.56 | 63.62 |
| 21–30 | 39 | 70 | 46.97 | 44.29 | 45.59 |
| >30 | 23 | 57 | 45.10 | 40.35 | 42.59 |
| ≤10 | 95 | 120 | 80.17 | 80.83 | 80.50 |

### Theo số bộ ba trong câu

| #bộ ba | Câu | Gold | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 184 | 184 | 60.85 | 70.11 | 65.15 |
| 2 | 102 | 204 | 67.76 | 60.78 | 64.08 |
| 3 | 23 | 69 | 76.79 | 62.32 | 68.80 |
| 4 | 14 | 56 | 65.91 | 51.79 | 58.00 |
| 5 | 2 | 10 | 16.67 | 10.00 | 12.50 |
| 6 | 3 | 18 | 50.00 | 33.33 | 40.00 |

### Ví dụ

**14lap-test-1** — tech support would not fix the problem unless I bought your plan for $ 150 plus .

- đáp án: `[('tech support', 'not fix', 'NEG')]`
- dự đoán: `[('tech support', 'problem', 'NEG')]`
- bỏ sót: `[('tech support', 'not fix', 'NEG')]` | thừa: `[('tech support', 'problem', 'NEG')]`

**14lap-test-7** — Sure it 's not light and slim but the features make up for it 100 % .

- đáp án: `[('features', 'not light and slim', 'POS')]`
- dự đoán: `[('features', 'make up', 'POS')]`
- bỏ sót: `[('features', 'not light and slim', 'POS')]` | thừa: `[('features', 'make up', 'POS')]`

**14lap-test-8** — I am pleased with the fast log on , speedy WiFi connection and the long battery life ( > 6 hrs ) .

- đáp án: `[('log on', 'pleased', 'POS'), ('log on', 'fast', 'POS'), ('wifi connection', 'pleased', 'POS'), ('wifi connection', 'speedy', 'POS'), ('battery life', 'pleased', 'POS'), ('battery life', 'long', 'POS')]`
- dự đoán: `[('log on', 'pleased', 'POS'), ('wifi connection', 'speedy', 'POS'), ('battery life', 'long', 'POS')]`
- bỏ sót: `[('log on', 'fast', 'POS'), ('wifi connection', 'pleased', 'POS'), ('battery life', 'pleased', 'POS')]` | thừa: `[]`

**14lap-test-15** — Strong build though which really adds to its durability .

- đáp án: `[('build', 'strong', 'POS'), ('durability', 'strong', 'POS')]`
- dự đoán: `[('build', 'strong', 'POS'), ('durability', 'adds', 'POS')]`
- bỏ sót: `[('durability', 'strong', 'POS')]` | thừa: `[('durability', 'adds', 'POS')]`

## 14lap seed1

- Bộ ba vàng: **541**, dự đoán: **510**, đúng: **332**
- **P = 65.10  R = 61.37  F1 = 63.18** (khớp tuyệt đối bộ ba)

| Loại lỗi | Số lượng | % lỗi phía đáp án |
|---|---:|---:|
| Sai biên | 82 | 39.2% |
| Sai cặp | 10 | 4.8% |
| Sai cảm xúc | 36 | 17.2% |
| Bỏ sót | 81 | 38.8% |
| Dự đoán thừa | 50 | — |
| Sai định dạng | 0 | — |

### Theo độ dài câu

| Độ dài (từ) | Câu | Gold | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| 11–20 | 171 | 294 | 63.67 | 60.20 | 61.89 |
| 21–30 | 39 | 70 | 56.06 | 52.86 | 54.41 |
| >30 | 23 | 57 | 46.81 | 38.60 | 42.31 |
| ≤10 | 95 | 120 | 80.67 | 80.00 | 80.33 |

### Theo số bộ ba trong câu

| #bộ ba | Câu | Gold | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 184 | 184 | 60.37 | 71.20 | 65.34 |
| 2 | 102 | 204 | 66.48 | 58.33 | 62.14 |
| 3 | 23 | 69 | 85.45 | 68.12 | 75.81 |
| 4 | 14 | 56 | 67.44 | 51.79 | 58.59 |
| 5 | 2 | 10 | 16.67 | 10.00 | 12.50 |
| 6 | 3 | 18 | 50.00 | 27.78 | 35.71 |

### Ví dụ

**14lap-test-3** — Did not enjoy the new Windows 8 and touchscreen functions .

- đáp án: `[('windows 8', 'not enjoy', 'NEG'), ('touchscreen functions', 'not enjoy', 'NEG')]`
- dự đoán: `[('windows 8', 'did not enjoy', 'NEG'), ('touchscreen functions', 'did not enjoy', 'NEG')]`
- bỏ sót: `[('windows 8', 'not enjoy', 'NEG'), ('touchscreen functions', 'not enjoy', 'NEG')]` | thừa: `[('windows 8', 'did not enjoy', 'NEG'), ('touchscreen functions', 'did not enjoy', 'NEG')]`

**14lap-test-6** — Works well , and I am extremely happy to be back to an apple OS .

- đáp án: `[('works', 'well', 'POS'), ('apple os', 'happy', 'POS')]`
- dự đoán: `[('works', 'well', 'POS'), ('works', 'happy', 'POS')]`
- bỏ sót: `[('apple os', 'happy', 'POS')]` | thừa: `[('works', 'happy', 'POS')]`

**14lap-test-7** — Sure it 's not light and slim but the features make up for it 100 % .

- đáp án: `[('features', 'not light and slim', 'POS')]`
- dự đoán: `[('features', 'make up', 'POS')]`
- bỏ sót: `[('features', 'not light and slim', 'POS')]` | thừa: `[('features', 'make up', 'POS')]`

**14lap-test-8** — I am pleased with the fast log on , speedy WiFi connection and the long battery life ( > 6 hrs ) .

- đáp án: `[('log on', 'pleased', 'POS'), ('log on', 'fast', 'POS'), ('wifi connection', 'pleased', 'POS'), ('wifi connection', 'speedy', 'POS'), ('battery life', 'pleased', 'POS'), ('battery life', 'long', 'POS')]`
- dự đoán: `[('log on', 'fast', 'POS'), ('wifi connection', 'speedy', 'POS'), ('battery life', 'long', 'POS')]`
- bỏ sót: `[('log on', 'pleased', 'POS'), ('wifi connection', 'pleased', 'POS'), ('battery life', 'pleased', 'POS')]` | thừa: `[]`

## 14res seed0

- Bộ ba vàng: **994**, dự đoán: **1010**, đúng: **749**
- **P = 74.16  R = 75.35  F1 = 74.75** (khớp tuyệt đối bộ ba)

| Loại lỗi | Số lượng | % lỗi phía đáp án |
|---|---:|---:|
| Sai biên | 105 | 42.9% |
| Sai cặp | 14 | 5.7% |
| Sai cảm xúc | 39 | 15.9% |
| Bỏ sót | 87 | 35.5% |
| Dự đoán thừa | 103 | — |
| Sai định dạng | 0 | — |

### Theo độ dài câu

| Độ dài (từ) | Câu | Gold | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| 11–20 | 240 | 477 | 72.46 | 76.10 | 74.23 |
| 21–30 | 85 | 209 | 68.25 | 68.90 | 68.57 |
| >30 | 34 | 115 | 66.67 | 60.87 | 63.64 |
| ≤10 | 133 | 193 | 89.12 | 89.12 | 89.12 |

### Theo số bộ ba trong câu

| #bộ ba | Câu | Gold | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 206 | 206 | 63.92 | 79.13 | 70.72 |
| 14 | 2 | 28 | 69.23 | 32.14 | 43.90 |
| 2 | 172 | 344 | 74.58 | 76.74 | 75.64 |
| 3 | 60 | 180 | 82.68 | 82.22 | 82.45 |
| 4 | 30 | 120 | 78.57 | 73.33 | 75.86 |
| 5 | 17 | 85 | 85.53 | 76.47 | 80.75 |
| 6 | 4 | 24 | 52.94 | 37.50 | 43.90 |
| 7 | 1 | 7 | 75.00 | 42.86 | 54.55 |

### Ví dụ

**14res-test-2** — Food is always fresh and hot ready to eat !

- đáp án: `[('food', 'fresh', 'POS'), ('food', 'hot', 'POS')]`
- dự đoán: `[('food', 'fresh', 'POS'), ('food', 'hot ready', 'POS')]`
- bỏ sót: `[('food', 'hot', 'POS')]` | thừa: `[('food', 'hot ready', 'POS')]`

**14res-test-7** — BEST spicy tuna roll , great asian salad .

- đáp án: `[('spicy tuna roll', 'best', 'POS'), ('asian salad', 'great', 'POS')]`
- dự đoán: `[('tuna roll', 'best', 'POS'), ('asian salad', 'great', 'POS')]`
- bỏ sót: `[('spicy tuna roll', 'best', 'POS')]` | thừa: `[('tuna roll', 'best', 'POS')]`

**14res-test-17** — The portions of the food that came out were mediocre .

- đáp án: `[('portions of the food', 'mediocre', 'NEU')]`
- dự đoán: `[('portions', 'mediocre', 'NEU')]`
- bỏ sót: `[('portions of the food', 'mediocre', 'NEU')]` | thừa: `[('portions', 'mediocre', 'NEU')]`

**14res-test-22** — Harumi Sushi has the freshest and most delicious array of sushi in NYC .

- đáp án: `[('array of sushi', 'freshest', 'POS'), ('array of sushi', 'most delicious', 'POS')]`
- dự đoán: `[('array of sushi', 'freshest', 'POS'), ('array of sushi', 'delicious', 'POS')]`
- bỏ sót: `[('array of sushi', 'most delicious', 'POS')]` | thừa: `[('array of sushi', 'delicious', 'POS')]`

## 14res seed1

- Bộ ba vàng: **994**, dự đoán: **1003**, đúng: **738**
- **P = 73.58  R = 74.25  F1 = 73.91** (khớp tuyệt đối bộ ba)

| Loại lỗi | Số lượng | % lỗi phía đáp án |
|---|---:|---:|
| Sai biên | 108 | 42.2% |
| Sai cặp | 20 | 7.8% |
| Sai cảm xúc | 32 | 12.5% |
| Bỏ sót | 96 | 37.5% |
| Dự đoán thừa | 105 | — |
| Sai định dạng | 0 | — |

### Theo độ dài câu

| Độ dài (từ) | Câu | Gold | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| 11–20 | 240 | 477 | 71.94 | 75.26 | 73.57 |
| 21–30 | 85 | 209 | 68.08 | 69.38 | 68.72 |
| >30 | 34 | 115 | 61.54 | 55.65 | 58.45 |
| ≤10 | 133 | 193 | 90.91 | 88.08 | 89.47 |

### Theo số bộ ba trong câu

| #bộ ba | Câu | Gold | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 206 | 206 | 65.59 | 78.64 | 71.52 |
| 14 | 2 | 28 | 53.85 | 25.00 | 34.15 |
| 2 | 172 | 344 | 74.79 | 76.74 | 75.75 |
| 3 | 60 | 180 | 79.23 | 80.56 | 79.89 |
| 4 | 30 | 120 | 75.44 | 71.67 | 73.50 |
| 5 | 17 | 85 | 83.10 | 69.41 | 75.64 |
| 6 | 4 | 24 | 61.11 | 45.83 | 52.38 |
| 7 | 1 | 7 | 100.00 | 57.14 | 72.73 |

### Ví dụ

**14res-test-7** — BEST spicy tuna roll , great asian salad .

- đáp án: `[('spicy tuna roll', 'best', 'POS'), ('asian salad', 'great', 'POS')]`
- dự đoán: `[('tuna roll', 'best', 'POS'), ('asian salad', 'great', 'POS')]`
- bỏ sót: `[('spicy tuna roll', 'best', 'POS')]` | thừa: `[('tuna roll', 'best', 'POS')]`

**14res-test-17** — The portions of the food that came out were mediocre .

- đáp án: `[('portions of the food', 'mediocre', 'NEU')]`
- dự đoán: `[('portions', 'mediocre', 'NEU')]`
- bỏ sót: `[('portions of the food', 'mediocre', 'NEU')]` | thừa: `[('portions', 'mediocre', 'NEU')]`

**14res-test-18** — From the beginning , we were met by friendly staff memebers , and the convenient parking at Chelsea Piers made it easy for us to get to the boat .

- đáp án: `[('staff memebers', 'friendly', 'POS'), ('parking', 'convenient', 'POS')]`
- dự đoán: `[('parking', 'convenient', 'POS'), ('staff', 'friendly', 'POS')]`
- bỏ sót: `[('staff memebers', 'friendly', 'POS')]` | thừa: `[('staff', 'friendly', 'POS')]`

**14res-test-22** — Harumi Sushi has the freshest and most delicious array of sushi in NYC .

- đáp án: `[('array of sushi', 'freshest', 'POS'), ('array of sushi', 'most delicious', 'POS')]`
- dự đoán: `[('array of sushi', 'freshest', 'POS'), ('array of sushi', 'delicious', 'POS')]`
- bỏ sót: `[('array of sushi', 'most delicious', 'POS')]` | thừa: `[('array of sushi', 'delicious', 'POS')]`

## 15res seed0

- Bộ ba vàng: **485**, dự đoán: **504**, đúng: **328**
- **P = 65.08  R = 67.63  F1 = 66.33** (khớp tuyệt đối bộ ba)

| Loại lỗi | Số lượng | % lỗi phía đáp án |
|---|---:|---:|
| Sai biên | 81 | 51.6% |
| Sai cặp | 7 | 4.5% |
| Sai cảm xúc | 30 | 19.1% |
| Bỏ sót | 39 | 24.8% |
| Dự đoán thừa | 58 | — |
| Sai định dạng | 0 | — |

### Theo độ dài câu

| Độ dài (từ) | Câu | Gold | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| 11–20 | 129 | 200 | 59.72 | 64.50 | 62.02 |
| 21–30 | 46 | 83 | 58.43 | 62.65 | 60.47 |
| >30 | 30 | 61 | 48.28 | 45.90 | 47.06 |
| ≤10 | 117 | 141 | 84.40 | 84.40 | 84.40 |

### Theo số bộ ba trong câu

| #bộ ba | Câu | Gold | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 210 | 210 | 60.78 | 73.81 | 66.67 |
| 2 | 77 | 154 | 65.10 | 62.99 | 64.03 |
| 3 | 22 | 66 | 75.00 | 63.64 | 68.85 |
| 4 | 10 | 40 | 71.88 | 57.50 | 63.89 |
| 5 | 3 | 15 | 91.67 | 73.33 | 81.48 |

### Ví dụ

**15res-test-3** — The pastas are incredible , the risottos ( particularly the sepia ) are fantastic and the braised rabbit is amazing .

- đáp án: `[('pastas', 'incredible', 'POS'), ('risottos', 'fantastic', 'POS'), ('sepia', 'fantastic', 'POS'), ('braised rabbit', 'amazing', 'POS')]`
- dự đoán: `[('pastas', 'incredible', 'POS'), ('risottos ( particularly the sepia )', 'fantastic', 'POS'), ('braised rabbit', 'amazing', 'POS')]`
- bỏ sót: `[('risottos', 'fantastic', 'POS'), ('sepia', 'fantastic', 'POS')]` | thừa: `[('risottos ( particularly the sepia )', 'fantastic', 'POS')]`

**15res-test-7** — An awesome organic dog , and a conscious eco friendly establishment .

- đáp án: `[('dog', 'organic', 'POS'), ('establishment', 'eco friendly', 'POS')]`
- dự đoán: `[('organic dog', 'awesome', 'POS'), ('establishment', 'conscious', 'POS'), ('establishment', 'eco friendly', 'POS')]`
- bỏ sót: `[('dog', 'organic', 'POS')]` | thừa: `[('organic dog', 'awesome', 'POS'), ('establishment', 'conscious', 'POS')]`

**15res-test-11** — I LOOOVE their eggplant pizza , as well as their pastas !

- đáp án: `[('eggplant pizza', 'looove', 'POS'), ('pastas', 'looove', 'POS')]`
- dự đoán: `[('eggplant pizza', 'looove', 'POS'), ('pastas', 'loove', 'POS')]`
- bỏ sót: `[('pastas', 'looove', 'POS')]` | thừa: `[('pastas', 'loove', 'POS')]`

**15res-test-12** — We had half/half pizza , mine was eggplant and my friend had the buffalo and it was sooo huge for a small size pizza !

- đáp án: `[('half/half pizza', 'huge', 'POS')]`
- dự đoán: `[('eggplant', 'half/half pizza', 'POS'), ('buffalo', 'huge', 'POS')]`
- bỏ sót: `[('half/half pizza', 'huge', 'POS')]` | thừa: `[('eggplant', 'half/half pizza', 'POS'), ('buffalo', 'huge', 'POS')]`

## 15res seed1

- Bộ ba vàng: **485**, dự đoán: **502**, đúng: **328**
- **P = 65.34  R = 67.63  F1 = 66.46** (khớp tuyệt đối bộ ba)

| Loại lỗi | Số lượng | % lỗi phía đáp án |
|---|---:|---:|
| Sai biên | 84 | 53.5% |
| Sai cặp | 5 | 3.2% |
| Sai cảm xúc | 31 | 19.7% |
| Bỏ sót | 37 | 23.6% |
| Dự đoán thừa | 54 | — |
| Sai định dạng | 0 | — |

### Theo độ dài câu

| Độ dài (từ) | Câu | Gold | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| 11–20 | 129 | 200 | 61.97 | 66.00 | 63.92 |
| 21–30 | 46 | 83 | 56.32 | 59.04 | 57.65 |
| >30 | 30 | 61 | 51.61 | 52.46 | 52.03 |
| ≤10 | 117 | 141 | 82.14 | 81.56 | 81.85 |

### Theo số bộ ba trong câu

| #bộ ba | Câu | Gold | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 210 | 210 | 59.68 | 70.48 | 64.63 |
| 2 | 77 | 154 | 68.97 | 64.94 | 66.89 |
| 3 | 22 | 66 | 72.58 | 68.18 | 70.31 |
| 4 | 10 | 40 | 68.57 | 60.00 | 64.00 |
| 5 | 3 | 15 | 91.67 | 73.33 | 81.48 |

### Ví dụ

**15res-test-3** — The pastas are incredible , the risottos ( particularly the sepia ) are fantastic and the braised rabbit is amazing .

- đáp án: `[('pastas', 'incredible', 'POS'), ('risottos', 'fantastic', 'POS'), ('sepia', 'fantastic', 'POS'), ('braised rabbit', 'amazing', 'POS')]`
- dự đoán: `[('pastas', 'incredible', 'POS'), ('risottos ( particularly the sepia )', 'fantastic', 'POS'), ('braised rabbit', 'amazing', 'POS')]`
- bỏ sót: `[('risottos', 'fantastic', 'POS'), ('sepia', 'fantastic', 'POS')]` | thừa: `[('risottos ( particularly the sepia )', 'fantastic', 'POS')]`

**15res-test-7** — An awesome organic dog , and a conscious eco friendly establishment .

- đáp án: `[('dog', 'organic', 'POS'), ('establishment', 'eco friendly', 'POS')]`
- dự đoán: `[('establishment', 'conscious', 'POS')]`
- bỏ sót: `[('dog', 'organic', 'POS'), ('establishment', 'eco friendly', 'POS')]` | thừa: `[('establishment', 'conscious', 'POS')]`

**15res-test-11** — I LOOOVE their eggplant pizza , as well as their pastas !

- đáp án: `[('eggplant pizza', 'looove', 'POS'), ('pastas', 'looove', 'POS')]`
- dự đoán: `[('eggplant pizza', 'looove', 'POS'), ('pastas', 'loove', 'POS')]`
- bỏ sót: `[('pastas', 'looove', 'POS')]` | thừa: `[('pastas', 'loove', 'POS')]`

**15res-test-12** — We had half/half pizza , mine was eggplant and my friend had the buffalo and it was sooo huge for a small size pizza !

- đáp án: `[('half/half pizza', 'huge', 'POS')]`
- dự đoán: `[('eggplant', 'huge', 'POS'), ('buffalo', 'huge', 'POS')]`
- bỏ sót: `[('half/half pizza', 'huge', 'POS')]` | thừa: `[('eggplant', 'huge', 'POS'), ('buffalo', 'huge', 'POS')]`

## 16res seed0

- Bộ ba vàng: **514**, dự đoán: **538**, đúng: **394**
- **P = 73.23  R = 76.65  F1 = 74.90** (khớp tuyệt đối bộ ba)

| Loại lỗi | Số lượng | % lỗi phía đáp án |
|---|---:|---:|
| Sai biên | 60 | 50.0% |
| Sai cặp | 5 | 4.2% |
| Sai cảm xúc | 25 | 20.8% |
| Bỏ sót | 30 | 25.0% |
| Dự đoán thừa | 54 | — |
| Sai định dạng | 0 | — |

### Theo độ dài câu

| Độ dài (từ) | Câu | Gold | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| 11–20 | 129 | 232 | 75.95 | 77.59 | 76.76 |
| 21–30 | 48 | 71 | 66.67 | 78.87 | 72.26 |
| >30 | 23 | 50 | 50.94 | 54.00 | 52.43 |
| ≤10 | 126 | 161 | 79.88 | 81.37 | 80.62 |

### Theo số bộ ba trong câu

| #bộ ba | Câu | Gold | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 192 | 192 | 67.11 | 79.69 | 72.86 |
| 2 | 97 | 194 | 74.33 | 71.65 | 72.97 |
| 3 | 23 | 69 | 87.88 | 84.06 | 85.93 |
| 4 | 11 | 44 | 75.61 | 70.45 | 72.94 |
| 5 | 3 | 15 | 81.25 | 86.67 | 83.87 |

### Ví dụ

**16res-test-1** — Not the biggest portions but adequate .

- đáp án: `[('portions', 'not the biggest', 'NEU'), ('portions', 'adequate', 'NEU')]`
- dự đoán: `[('portions', 'not the biggest', 'NEU'), ('portions', 'adequate', 'POS')]`
- bỏ sót: `[('portions', 'adequate', 'NEU')]` | thừa: `[('portions', 'adequate', 'POS')]`

**16res-test-7** — This is a great place to get a delicious meal .

- đáp án: `[('meal', 'delicious', 'POS')]`
- dự đoán: `[('place', 'great', 'POS')]`
- bỏ sót: `[('meal', 'delicious', 'POS')]` | thừa: `[('place', 'great', 'POS')]`

**16res-test-12** — Worst Service I Ever Had

- đáp án: `[('service', 'worst', 'POS')]`
- dự đoán: `[('service', 'worst', 'NEG')]`
- bỏ sót: `[('service', 'worst', 'POS')]` | thừa: `[('service', 'worst', 'NEG')]`

**16res-test-24** — – In a age of incremental cost cutting in restaurants , its nice to see a place that bucks that trend , and just plain delivers high quality food and good service , period .

- đáp án: `[('food', 'high quality', 'POS'), ('service', 'good', 'POS')]`
- dự đoán: `[('food', 'bucks that trend', 'POS'), ('service', 'good', 'POS')]`
- bỏ sót: `[('food', 'high quality', 'POS')]` | thừa: `[('food', 'bucks that trend', 'POS')]`

## 16res seed1

- Bộ ba vàng: **514**, dự đoán: **542**, đúng: **392**
- **P = 72.32  R = 76.26  F1 = 74.24** (khớp tuyệt đối bộ ba)

| Loại lỗi | Số lượng | % lỗi phía đáp án |
|---|---:|---:|
| Sai biên | 55 | 45.1% |
| Sai cặp | 7 | 5.7% |
| Sai cảm xúc | 27 | 22.1% |
| Bỏ sót | 33 | 27.0% |
| Dự đoán thừa | 61 | — |
| Sai định dạng | 0 | — |

### Theo độ dài câu

| Độ dài (từ) | Câu | Gold | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| 11–20 | 129 | 232 | 74.79 | 75.43 | 75.11 |
| 21–30 | 48 | 71 | 62.50 | 77.46 | 69.18 |
| >30 | 23 | 50 | 49.12 | 56.00 | 52.34 |
| ≤10 | 126 | 161 | 82.21 | 83.23 | 82.72 |

### Theo số bộ ba trong câu

| #bộ ba | Câu | Gold | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 192 | 192 | 65.22 | 78.12 | 71.09 |
| 2 | 97 | 194 | 75.81 | 72.68 | 74.21 |
| 3 | 23 | 69 | 85.51 | 85.51 | 85.51 |
| 4 | 11 | 44 | 73.17 | 68.18 | 70.59 |
| 5 | 3 | 15 | 75.00 | 80.00 | 77.42 |

### Ví dụ

**16res-test-7** — This is a great place to get a delicious meal .

- đáp án: `[('meal', 'delicious', 'POS')]`
- dự đoán: `[('place', 'great', 'POS')]`
- bỏ sót: `[('meal', 'delicious', 'POS')]` | thừa: `[('place', 'great', 'POS')]`

**16res-test-12** — Worst Service I Ever Had

- đáp án: `[('service', 'worst', 'POS')]`
- dự đoán: `[('service', 'worst', 'NEG')]`
- bỏ sót: `[('service', 'worst', 'POS')]` | thừa: `[('service', 'worst', 'NEG')]`

**16res-test-15** — I complained to the manager , but he was not even apologetic .

- đáp án: `[('manager', 'not even apologetic', 'NEG')]`
- dự đoán: `[('manager', 'complained', 'NEG')]`
- bỏ sót: `[('manager', 'not even apologetic', 'NEG')]` | thừa: `[('manager', 'complained', 'NEG')]`

**16res-test-25** — This is the place to relax and enjoy the finest quality food the industry can offer .

- đáp án: `[('place', 'relax', 'POS'), ('food', 'finest quality', 'POS')]`
- dự đoán: `[('food', 'finest quality', 'POS')]`
- bỏ sót: `[('place', 'relax', 'POS')]` | thừa: `[]`
