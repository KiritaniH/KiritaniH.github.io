import argparse
import json
from collections import defaultdict
from typing import List, Dict

def compute_length_score(text: str, min_len: int = 40, max_len: int = 200) -> float:
    words = text.split()
    n = len(words)
    if n == 0:
        return 0.0

    target = (min_len + max_len) / 2
    max_diff = target - min_len
    diff = abs(n - target)
    score = max(0.0, 1.0 - diff / max_diff)
    return score


def compute_repetition_score(text: str) -> float:
    tokens = text.lower().split()
    n = 3
    seen = set()
    repeats = 0
    for i in range(len(tokens) - n):
        gram = tuple(tokens[i:i + n])
        if gram in seen:
            repeats += 1
        else:
            seen.add(gram)
    score = max(0.0, 1.0 - repeats / 5.0)
    return score


def compute_completeness_score(text: str) -> float:
    text = text.strip()
    if len(text) < 20:
        return 0.2
    if text.endswith(("。", ".", "！", "!", "？", "?")):
        return 1.0
    return 0.7


def compute_quality(text: str) -> float:
    l = compute_length_score(text)
    r = compute_repetition_score(text)
    c = compute_completeness_score(text)
    return 0.4 * l + 0.3 * r + 0.3 * c

def load_sft_data(path: str) -> List[Dict]:
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
            out = obj.get("output", "").strip()
            if not ins or not out:
                continue
            meta = obj.get("meta", {})
            data.append({"instruction": ins, "output": out, "meta": meta})
    print(f"Loaded {len(data)} raw SFT samples from {path}")
    return data

def quality_filter(
    data: List[Dict],
    min_quality: float = 0.7,
) -> List[Dict]:
    kept = []
    for item in data:
        out = item["output"]
        q = compute_quality(out)
        item["quality"] = q
        if q >= min_quality:
            kept.append(item)
    print(f"After quality filter (min_quality={min_quality}): {len(kept)} / {len(data)} samples kept.")
    return kept

def coverage_filter(
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
        items_sorted = sorted(items, key=lambda x: x.get("quality", 0.0), reverse=True)
        selected = items_sorted[:per_bucket_max]
        filtered.extend(selected)
        print(f"Bucket {key}: {len(items)} -> {len(selected)} kept.")

    print(f"After coverage filter (per_bucket_max={per_bucket_max}): {len(filtered)} samples kept.")
    return filtered
def save_sft_data(path: str, data: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            obj = {
                "instruction": item["instruction"],
                "input": "",
                "output": item["output"],
                "meta": item.get("meta", {}),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} filtered SFT samples to {path}")

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
        "--min_quality",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--per_bucket_max",
        type=int,
        default=300,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data = load_sft_data(args.input)
    data_q = quality_filter(data, min_quality=args.min_quality)
    data_final = coverage_filter(data_q, per_bucket_max=args.per_bucket_max)
    save_sft_data(args.output, data_final)
