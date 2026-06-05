import argparse
import json
import re
from http import HTTPStatus
from typing import List, Dict

import dashscope
from dashscope import Generation
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def compute_reward(text: str) -> float:
    l = compute_length_score(text)
    r = compute_repetition_score(text)
    c = compute_completeness_score(text)
    return 0.4 * l + 0.3 * r + 0.3 * c

def build_chat_prompt(instruction: str) -> str:
    system_prompt = (
        "你是一名擅长讲解线性代数的大学教师，"
        "需要为学生提供清晰、循序渐进、逻辑严谨的中文解答。"
    )
    full_prompt = (
        f"[SYSTEM]\n{system_prompt}\n\n"
        f"[USER]\n{instruction}\n\n"
        f"[ASSISTANT]\n"
    )
    return full_prompt


def call_qwen(model_name: str, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    response = Generation.call(
        model=model_name,
        prompt=prompt,
        max_length=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"status={response.status_code}, code={response.code}, msg={response.message}"
        )
    answer = str(response.output).strip()
    return answer


def load_instructions_from_sft(sft_path: str, max_num: int) -> List[Dict]:
    data = []
    with open(sft_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(data) >= max_num:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ins = obj.get("instruction")
            if not ins:
                continue
            meta = obj.get("meta", {})
            data.append({"instruction": ins, "meta": meta})
    print(f"Loaded {len(data)} instructions from {sft_path}")
    return data

def _generate_one_pref_sample(
    idx: int,
    item: Dict,
    model_name: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    instruction = item["instruction"]
    meta = item.get("meta", {})

    prompt = build_chat_prompt(instruction)
    candidates = []
    scores = []

    for j in range(k):
        try:
            answer = call_qwen(
                model_name=model_name,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as e:
            print(f"[{idx + 1}] 第 {j + 1} 个候选生成失败：{e}")
            continue

        candidates.append(answer)
        scores.append(compute_reward(answer))

    if len(candidates) < 2:
        print(f"[{idx + 1}] 候选回答不足 2 个，跳过。")
        return None

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    worst_idx = min(range(len(scores)), key=lambda i: scores[i])

    chosen = candidates[best_idx]
    rejected = candidates[worst_idx]

    sample = {
        "instruction": instruction,
        "response_chosen": chosen,
        "response_rejected": rejected,
        "score_chosen": scores[best_idx],
        "score_rejected": scores[worst_idx],
        "meta": meta,
    }
    return sample


def generate_preference_dataset(
    model_name: str,
    sft_path: str,
    num_pairs: int,
    k: int,
    outfile: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    num_workers: int = 4,
):
    sft_data = load_instructions_from_sft(sft_path, max_num=num_pairs)

    print(f"Using DashScope model: {model_name}")
    print(f"Generating {len(sft_data)} preference pairs with k={k}, workers={num_workers}")

    finished = 0
    with open(outfile, "w", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _generate_one_pref_sample,
                    idx,
                    item,
                    model_name,
                    k,
                    max_new_tokens,
                    temperature,
                    top_p,
                ): idx
                for idx, item in enumerate(sft_data)
            }

            for future in as_completed(futures):
                idx = futures[future]
                sample = future.result()
                if sample is None:
                    continue
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                finished += 1

                if finished == 1 or finished % 10 == 0:
                    print(
                        f"[{finished}/{num_pairs}] "
                        f"Generated pref sample for topic={sample.get('meta', {}).get('topic', 'N/A')}, "
                        f"diff={sample.get('meta', {}).get('difficulty', 'N/A')}"
                    )

    print(f"\nDone! Saved {finished} preference pairs to {outfile}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sft_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pref_linear_algebra.jsonl",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_preference_dataset(
        model_name=args.model,
        sft_path=args.sft_file,
        num_pairs=args.num_pairs,
        k=args.k,
        outfile=args.output,
        num_workers=args.num_workers,
    )
