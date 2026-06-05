import os
import re
import json
import math
import random
import hashlib
from collections import defaultdict, Counter

from transformers import AutoTokenizer

DATA_PATH = "./data/datamind_12k.json"
MODEL_PATH = "./model"
OUT_DIR = "./output"

TRAIN_SIZE = 2000
VAL_SIZE = 500
TOTAL_SIZE = TRAIN_SIZE + VAL_SIZE
RANDOM_SEED = 42

MAX_TOKENS = 4096

MIN_QUALITY_SCORE = 6.0
MIN_COMPLEXITY_SCORE = 5.0


def set_seed(seed: int):
    random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON object to be a list.")

    return data


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def short_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def valid_messages(sample):
    if not isinstance(sample, dict):
        return None

    messages = sample.get("messages")
    if not isinstance(messages, list):
        return None

    clean = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        clean.append({"role": role, "content": content.strip()})

    roles = [m["role"] for m in clean]
    if "user" not in roles or "assistant" not in roles:
        return None

    return clean


def non_system_messages(messages):
    return [m for m in messages if m["role"] != "system"]


def original_user_question(messages):
    for m in messages:
        if m["role"] == "user":
            c = m["content"]
            if not c.strip().startswith("<interpreter>"):
                return c
    return ""


def assistant_text(messages):
    return "\n".join(m["content"] for m in messages if m["role"] == "assistant")


def non_system_text(messages):
    return "\n".join(m["content"] for m in non_system_messages(messages))


THINK_PAT = re.compile(r"<think>.*?</think>", re.I | re.S)
CODE_PAT = re.compile(r"<code>.*?</code>|```python|```sql|execute_sql|get_db_info|import pandas|import numpy|pd\.|np\.", re.I | re.S)
ANSWER_PAT = re.compile(r"<answer>.*?</answer>", re.I | re.S)
INTERPRETER_PAT = re.compile(r"<interpreter>.*?</interpreter>", re.I | re.S)
ERROR_PAT = re.compile(r"error|traceback|exception|failed|no such table|syntax error", re.I)

SQL_PAT = re.compile(r"\bsql\b|sqlite|select |from |where |group by|join |execute_sql|get_db_info", re.I)
CSV_PAT = re.compile(r"\bcsv\b|excel|xlsx|pandas|dataframe|read_csv|read_excel", re.I)
ML_PAT = re.compile(r"classification|regression|accuracy|f1|auc|train_test_split|sklearn|model|predict", re.I)
STAT_PAT = re.compile(r"mean|median|variance|standard deviation|correlation|hypothesis|p-value|anova|statistical", re.I)
MATH_MODEL_PAT = re.compile(r"optimization|constraint|objective|linear programming|modeling|forecast|time series", re.I)

BAD_PAT = re.compile(r"as an ai|i cannot|i can't|sorry|not enough information", re.I)

def quality_score(messages):

    text = non_system_text(messages)
    a_text = assistant_text(messages)
    q = original_user_question(messages)

    score = 0.0

    if len(q) >= 20:
        score += 1.0

    if len(a_text) >= 100:
        score += 1.0

    if THINK_PAT.search(a_text):
        score += 0.8
    if CODE_PAT.search(a_text):
        score += 0.8
    if ANSWER_PAT.search(a_text):
        score += 1.0

    if INTERPRETER_PAT.search(text):
        score += 0.7

    if BAD_PAT.search(a_text):
        score -= 1.0

    last_assistant = ""
    for m in reversed(messages):
        if m["role"] == "assistant":
            last_assistant = m["content"]
            break

    if ANSWER_PAT.search(last_assistant):
        score += 0.7

    return max(score, 0.0)


def complexity_score(messages):

    ns_msgs = non_system_messages(messages)
    text = "\n".join(m["content"] for m in ns_msgs)
    a_text = assistant_text(messages)

    score = 0.0

    num_user = sum(1 for m in ns_msgs if m["role"] == "user")
    num_assistant = sum(1 for m in ns_msgs if m["role"] == "assistant")
    num_interpreter = len(INTERPRETER_PAT.findall(text))

    if num_user >= 2 and num_assistant >= 2:
        score += 0.8
    if num_user >= 3 and num_assistant >= 3:
        score += 0.8

    if num_interpreter >= 1:
        score += 0.7
    if num_interpreter >= 2:
        score += 0.7

    if THINK_PAT.search(a_text):
        score += 0.7
    if CODE_PAT.search(a_text):
        score += 1.0

    if ERROR_PAT.search(text):
        score += 0.7

    if SQL_PAT.search(text):
        score += 0.7
    if CSV_PAT.search(text):
        score += 0.7
    if ML_PAT.search(text):
        score += 0.7
    if STAT_PAT.search(text):
        score += 0.5
    if MATH_MODEL_PAT.search(text):
        score += 0.5

    n_chars = len(text)
    if n_chars >= 800:
        score += 0.5
    if n_chars >= 1500:
        score += 0.5
    if n_chars >= 3000:
        score += 0.5

    return score


def detect_bucket(messages):

    text = non_system_text(messages).lower()

    scores = {
        "sql_database": len(SQL_PAT.findall(text)),
        "csv_excel_analysis": len(CSV_PAT.findall(text)),
        "machine_learning": len(ML_PAT.findall(text)),
        "statistics": len(STAT_PAT.findall(text)),
        "math_modeling": len(MATH_MODEL_PAT.findall(text)),
    }

    best = max(scores.items(), key=lambda x: x[1])
    if best[1] == 0:
        return "general_data_analysis"
    return best[0]


def dedup_key(messages):

    q = original_user_question(messages)
    a = assistant_text(messages)
    key = normalize_text(q) + " ||| " + normalize_text(a[:1500])
    return short_hash(key)

def count_tokens(tokenizer, messages):
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    except Exception:
        rendered = ""
        for m in messages:
            rendered += f"{m['role']}:\n{m['content']}\n\n"

    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    return len(ids)

def main():
    set_seed(RANDOM_SEED)
    ensure_dir(OUT_DIR)

    print(f"[1] Loading data from {DATA_PATH}")
    raw = load_data(DATA_PATH)
    print(f"Raw samples: {len(raw)}")

    print(f"[2] Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    selected_candidates = []
    parse_fail = 0
    duplicate_drop = 0
    too_long_drop = 0
    low_score_drop = 0

    seen = set()

    for i, sample in enumerate(raw):
        messages = valid_messages(sample)
        if messages is None:
            parse_fail += 1
            continue

        q_score = quality_score(messages)
        c_score = complexity_score(messages)

        if q_score < MIN_QUALITY_SCORE or c_score < MIN_COMPLEXITY_SCORE:
            low_score_drop += 1
            continue

        dk = dedup_key(messages)
        if dk in seen:
            duplicate_drop += 1
            continue
        seen.add(dk)

        n_tokens = count_tokens(tokenizer, messages)
        if n_tokens > MAX_TOKENS:
            too_long_drop += 1
            continue

        bucket = detect_bucket(messages)

        final_score = (
            1.5 * q_score
            + 1.2 * c_score
            + 0.1 * math.log(max(n_tokens, 2))
        )

        selected_candidates.append({
            "id": sample.get("id", f"sample_{i}"),
            "messages": messages,
            "quality_score": round(q_score, 4),
            "complexity_score": round(c_score, 4),
            "final_score": round(final_score, 4),
            "bucket": bucket,
            "n_tokens": n_tokens,
        })

    print("[3] Filtering summary")
    print(f"  parse_fail: {parse_fail}")
    print(f"  low_score_drop: {low_score_drop}")
    print(f"  duplicate_drop: {duplicate_drop}")
    print(f"  too_long_drop: {too_long_drop}")
    print(f"  remaining: {len(selected_candidates)}")

    if len(selected_candidates) < TOTAL_SIZE:
        raise ValueError(
            f"Only {len(selected_candidates)} samples left, less than {TOTAL_SIZE}. "
            "Try lowering MIN_QUALITY_SCORE, MIN_COMPLEXITY_SCORE, or increasing MAX_TOKENS."
        )

    by_bucket = defaultdict(list)
    for x in selected_candidates:
        by_bucket[x["bucket"]].append(x)

    for k in by_bucket:
        by_bucket[k].sort(key=lambda x: x["final_score"], reverse=True)

    print("[4] Bucket distribution")
    for k, v in sorted(by_bucket.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {k}: {len(v)}")

    # 按桶分层取样，避免全部来自 SQL 或 CSV
    total_available = sum(len(v) for v in by_bucket.values())
    quota = {}

    for k, v in by_bucket.items():
        q = round(TOTAL_SIZE * len(v) / total_available)
        q = max(1, q)
        q = min(q, len(v))
        quota[k] = q

    # 调整 quota 到 TOTAL_SIZE
    while sum(quota.values()) > TOTAL_SIZE:
        k = max(quota, key=lambda x: quota[x])
        if quota[k] > 1:
            quota[k] -= 1
        else:
            break

    while sum(quota.values()) < TOTAL_SIZE:
        candidates = [k for k in quota if quota[k] < len(by_bucket[k])]
        if not candidates:
            break
        k = max(candidates, key=lambda x: len(by_bucket[x]) - quota[x])
        quota[k] += 1

    chosen = []
    for k, q in quota.items():
        chosen.extend(by_bucket[k][:q])

    if len(chosen) < TOTAL_SIZE:
        chosen_ids = {x["id"] for x in chosen}
        rest = [
            x for x in sorted(selected_candidates, key=lambda z: z["final_score"], reverse=True)
            if x["id"] not in chosen_ids
        ]
        chosen.extend(rest[:TOTAL_SIZE - len(chosen)])

    chosen = chosen[:TOTAL_SIZE]
    random.shuffle(chosen)

    val_data = chosen[:VAL_SIZE]
    train_data = chosen[VAL_SIZE:]

    train_out = [{"messages": x["messages"]} for x in train_data]
    val_out = [{"messages": x["messages"]} for x in val_data]

    # json
    with open(os.path.join(OUT_DIR, "train_qwen.json"), "w", encoding="utf-8") as f:
        json.dump(train_out, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "val_qwen.json"), "w", encoding="utf-8") as f:
        json.dump(val_out, f, ensure_ascii=False, indent=2)

    # jsonl
    with open(os.path.join(OUT_DIR, "train_qwen.jsonl"), "w", encoding="utf-8") as f:
        for x in train_out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with open(os.path.join(OUT_DIR, "val_qwen.jsonl"), "w", encoding="utf-8") as f:
        for x in val_out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    report = {
        "raw_samples": len(raw),
        "parse_fail": parse_fail,
        "low_score_drop": low_score_drop,
        "duplicate_drop": duplicate_drop,
        "too_long_drop": too_long_drop,
        "candidate_count": len(selected_candidates),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "thresholds": {
            "MIN_QUALITY_SCORE": MIN_QUALITY_SCORE,
            "MIN_COMPLEXITY_SCORE": MIN_COMPLEXITY_SCORE,
            "MAX_TOKENS": MAX_TOKENS,
        },
        "bucket_distribution_candidates": {
            k: len(v) for k, v in by_bucket.items()
        },
        "bucket_distribution_train": dict(Counter(x["bucket"] for x in train_data)),
        "bucket_distribution_val": dict(Counter(x["bucket"] for x in val_data)),
        "top_20_selected_meta": [
            {
                "id": x["id"],
                "quality_score": x["quality_score"],
                "complexity_score": x["complexity_score"],
                "final_score": x["final_score"],
                "bucket": x["bucket"],
                "n_tokens": x["n_tokens"],
            }
            for x in sorted(chosen, key=lambda z: z["final_score"], reverse=True)[:20]
        ]
    }

    with open(os.path.join(OUT_DIR, "selection_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[5] Done")
    print(f"Train: {os.path.join(OUT_DIR, 'train_qwen.json')}")
    print(f"Val:   {os.path.join(OUT_DIR, 'val_qwen.json')}")
    print(f"Report:{os.path.join(OUT_DIR, 'selection_report.json')}")


if __name__ == "__main__":
    main()