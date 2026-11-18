"""Kiểm thử script đánh giá chung. Chạy từ thư mục gốc:  python -m unittest discover eval"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aste_eval as E


class TestGold(unittest.TestCase):
    def test_sizes(self):
        # 14lap test có 543 bộ ba trong file gốc, trong đó 2 bộ ba bị ghi lặp ở dòng 281.
        sizes = {"14lap": (328, 541), "14res": (492, 994), "15res": (322, 485), "16res": (326, 514)}
        for tap, (n_sent, n_tri) in sizes.items():
            gold = E.read_gold(tap)
            self.assertEqual(len(gold), n_sent)
            self.assertEqual(sum(len(g["triplets"]) for g in gold), n_tri)

    def test_first_line(self):
        g = E.read_gold("14lap")[0]
        self.assertEqual(g["id"], "14lap-test-0")
        self.assertEqual(g["tokens"][:2], ["Boot", "time"])
        self.assertEqual(g["triplets"], [(0, 1, 4, 4, "POS")])

    def test_id_lookup_with_duplicates(self):
        gold = E.read_gold("16res")
        lookup = E.IdLookup("16res")
        ids = [lookup.get(" ".join(g["tokens"])) for g in gold]
        self.assertEqual(ids, [g["id"] for g in gold])
        with self.assertRaises(KeyError):
            lookup.get(gold[0]["tokens"])
        with self.assertRaises(KeyError):
            E.IdLookup("16res").get("câu này không có trong tập")


class TestScore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "seed0.jsonl")
        self.gold = E.read_gold("14lap")

    def write(self, records):
        with open(self.path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def gold_records(self):
        return [E.make_record(g, g["triplets"]) for g in self.gold]

    def test_gold_scores_100(self):
        E.write_preds(self.path, "14lap", [g["triplets"] for g in self.gold])
        res = E.evaluate_file(self.path)
        self.assertEqual(res["tap"], "14lap")
        self.assertAlmostEqual(res["test_f1"], 100.0)
        self.assertAlmostEqual(res["test_f1_chi_so"], 100.0)

    def test_empty_scores_0(self):
        E.write_preds(self.path, "14lap", [[] for _ in self.gold])
        res = E.evaluate_file(self.path)
        self.assertEqual((res["test_p"], res["test_r"], res["test_f1"]), (0.0, 0.0, 0.0))

    def test_known_counts(self):
        # Giữ đáp án của câu 0, câu 1 sai cảm xúc, các câu còn lại để trống.
        preds = [[] for _ in self.gold]
        preds[0] = self.gold[0]["triplets"]
        a, b, c, d, s = self.gold[1]["triplets"][0]
        preds[1] = [(a, b, c, d, "POS" if s != "POS" else "NEG")]
        E.write_preds(self.path, "14lap", preds)
        res = E.evaluate_file(self.path)
        self.assertAlmostEqual(res["test_p"], 50.0)
        self.assertAlmostEqual(res["test_r"], 100.0 / 541)
        self.assertAlmostEqual(res["test_p_chi_so"], 50.0)

    def test_text_only(self):
        preds = []
        for g in self.gold:
            t = g["tokens"]
            preds.append([(" ".join(t[a:b + 1]).upper(), " ".join(t[c:d + 1]), s)
                          for a, b, c, d, s in g["triplets"]])
        E.write_preds(self.path, "14lap", preds, text_only=True)
        res = E.evaluate_file(self.path)
        self.assertAlmostEqual(res["test_f1"], 100.0)
        self.assertIsNone(res["test_f1_chi_so"])

    def test_any_order_accepted(self):
        self.write(list(reversed(self.gold_records())))
        self.assertAlmostEqual(E.evaluate_file(self.path)["test_f1"], 100.0)

    def assert_rejected(self, records):
        self.write(records)
        with self.assertRaises(E.FormatError):
            E.evaluate_file(self.path)

    def test_missing_sentence(self):
        self.assert_rejected(self.gold_records()[:-1])

    def test_duplicate_id(self):
        recs = self.gold_records()
        recs[1] = dict(recs[1], id=recs[0]["id"])
        self.assert_rejected(recs)

    def test_wrong_tokens(self):
        recs = self.gold_records()
        recs[0]["tokens"] = [t.lower() for t in recs[0]["tokens"]]
        self.assert_rejected(recs)

    def test_bad_sentiment(self):
        recs = self.gold_records()
        recs[0]["pred"][0]["sentiment"] = "positive"
        self.assert_rejected(recs)

    def test_span_out_of_range(self):
        recs = self.gold_records()
        recs[0]["pred"][0]["aspect"] = [0, 999]
        self.assert_rejected(recs)

    def test_text_not_matching_index(self):
        recs = self.gold_records()
        recs[0]["pred"][0]["aspect_text"] = "boot"
        self.assert_rejected(recs)

    def test_duplicate_triplet(self):
        recs = self.gold_records()
        recs[0]["pred"].append(dict(recs[0]["pred"][0]))
        self.assert_rejected(recs)

    def test_half_range(self):
        mean, half = E.mean_and_half_range([58.96, 59.80])
        self.assertAlmostEqual(mean, 59.38)
        self.assertAlmostEqual(half, 0.42)


if __name__ == "__main__":
    unittest.main()
