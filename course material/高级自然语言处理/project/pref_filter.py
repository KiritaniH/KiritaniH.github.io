import argparse
import json
from collections import defaultdict
from typing import List, Dict


def load_pref_data(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            ins = obj.get("instruction", "").strip()
            rc = obj.get("response_chosen", "").strip()
            rr = obj.get("response_rejected", "").strip()
            sc = obj.get("score_chosen", None)
            sr = obj.get("score_rejected", None)
            if not ins or not rc or not rr:
                continue
            if sc is None or sr is None:
                continue

            meta = obj.get("meta", {})
            data.append({
                "instruction": ins,
                "response_chosen": rc,
                "response_rejected": rr,
                "score_chosen": sc,
                "score_rejected": sr,
                "meta": meta,
            })

    print(f"Loaded {len(data)} raw preference samples from {path}")
    return data

def confidence_filter(
    data: List[Dict],
    min_gap: float = 0.2,
    min_chosen_score: float = 0.6,
) -> List[Dict]:
    kept = []
    for item in data:
        sc = float(item["score_chosen"])
        sr = float(item["score_rejected"])
        gap = sc - sr
        item["gap"] = gap 

        if gap >= min_gap and sc >= min_chosen_score:
            kept.append(item)

    print(
        f"After confidence filter (min_gap={min_gap}, "
        f"min_chosen_score={min_chosen_score}): "
        f"{len(kept)} / {len(data)} samples kept."
    )
    return kept
def coverage_filter_pref(
    data: List[Dict],
    per_bucket_max: int = 300,
) -> List[Dict]:
    buckets = defaultdict(list)

    for item in data:
        meta = item.get("meta", {})
        topic = meta.get("topic", "UNKNOWN_TOPIC")
        diff = meta.get("difficulty", "UNKNOWN_DIFF")
        key = (topic, diff)
        buckets[key].append(item)

    print(f"Found {len(buckets)} buckets by (topic, difficulty).")

    filtered = []
    for key, items in buckets.items():
        topic, diff = key
        items_sorted = sorted(items, key=lambda x: x.get("gap", 0.0), reverse=True)
        selected = items_sorted[:per_bucket_max]
        filtered.extend(selected)
        print(f"Bucket {key}: {len(items)} -> {len(selected)} kept.")

    print(f"After coverage filter (per_bucket_max={per_bucket_max}): {len(filtered)} samples kept.")
    return filtered

def save_pref_data(path: str, data: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            obj = {
                "instruction": item["instruction"],
                "response_chosen": item["response_chosen"],
                "response_rejected": item["response_rejected"],
                "score_chosen": float(item["score_chosen"]),
                "score_rejected": float(item["score_rejected"]),
                "meta": item.get("meta", {}),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} filtered preference samples to {path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--min_gap",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--min_chosen_score",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--per_bucket_max",
        type=int,
        default=300,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data = load_pref_data(args.input)
    data_q = confidence_filter(
        data,
        min_gap=args.min_gap,
        min_chosen_score=args.min_chosen_score,
    )
    data_final = coverage_filter_pref(
        data_q,
        per_bucket_max=args.per_bucket_max,
    )
    save_pref_data(args.output, data_final)
