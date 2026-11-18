"""Phần nhóm thêm vào GTS, dùng chung cho BertModel và NNModel: đặt seed, ghi file dự đoán theo
format chung của nhóm và ghi các số đo chi phí (docs/quy-uoc-chung.md, mục 4 và 6)."""

import json
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'eval'))

# tên tập của GTS -> tên tập của nhóm
DATASET_NAMES = {'lap14': '14lap', 'res14': '14res', 'res15': '15res', 'res16': '16res'}
# nhãn trên lưới của GTS -> nhãn cảm xúc của nhóm
ID2SENTIMENT = {3: 'NEG', 4: 'NEU', 5: 'POS'}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_name(args):
    return '{}_seed{}'.format(args.dataset, args.seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def save_predictions(args, sentence_packs, sentence_ids, triplets):
    """sentence_ids[i] là trường `id` của câu thứ i mà mô hình đã dự đoán, triplets[i] là các bộ ba
    [al, ar, pl, pr, nhãn] của câu đó. Câu ở vị trí k trong test.json là dòng k của tập test gốc."""
    from aste_eval import write_preds

    position = {pack['id']: k for k, pack in enumerate(sentence_packs)}
    assert len(position) == len(sentence_packs) == len(sentence_ids)
    preds = [None] * len(sentence_packs)
    for sentence_id, sentence_triplets in zip(sentence_ids, triplets):
        preds[position[sentence_id]] = [(al, ar, pl, pr, ID2SENTIMENT[s])
                                        for al, ar, pl, pr, s in sentence_triplets]
    write_preds(args.pred_file, DATASET_NAMES[args.dataset], preds)
    print('predictions saved to ' + args.pred_file)


def save_stats(args, stats):
    """Ghi thêm các số đo vào file thống kê của lần chạy (train và test ghi hai lần vào cùng file)."""
    if os.path.exists(args.stats_file):
        stats = dict(json.load(open(args.stats_file)), **stats)
    os.makedirs(os.path.dirname(os.path.abspath(args.stats_file)), exist_ok=True)
    json.dump(stats, open(args.stats_file, 'w'), indent=1)
    print('stats: ' + json.dumps(stats))
