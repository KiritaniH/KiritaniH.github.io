import argparse
import json
import random
from typing import List, Dict
from http import HTTPStatus
import dashscope
from dashscope import Generation
from concurrent.futures import ThreadPoolExecutor, as_completed

LINEAR_ALGEBRA_TOPICS = [
    "向量与向量空间",
    "线性无关与基",
    "矩阵与线性变换",
    "秩与零空间",
    "行列式及其性质",
    "特征值与特征向量",
    "对角化",
    "正交与正交投影",
    "内积空间与正交基",
    "奇异值分解（SVD）",
    "矩阵的逆与伪逆",
]


DIFFICULTY_LEVELS = [
    ("easy",   "给大一刚学线代的学生做一个入门讲解"),
    ("medium", "面向已经学完一学期线代的学生，要求适当有推导和例子"),
    ("hard",   "面向想继续学习高等代数/矩阵分析的学生，要求有一定深度和数学严谨性"),
]


def build_instruction(topic: str, difficulty_tag: str, difficulty_desc: str) -> str:
    templates = [
        "你是一名大学线性代数老师，请根据{difficulty_desc}，用中文讲解“{topic}”这一知识点，并举一个具体例子。整体回答控制在200到300字以内。",
        "围绕“{topic}”，请你根据{difficulty_desc}，设计一段教学讲解，要求包括：直观解释、一个例题、以及该知识点在实际问题中的一个简单应用。整体回答控制在200到300字以内。",
        "请根据{difficulty_desc}，为学生编写一段关于“{topic}”的学习笔记，包括：概念、关键性质、常见错误，以及一个小练习题（附参考答案）。整体回答控制在200到300字以内。",
        "假设学生正在准备线性代数考试，请以{difficulty_desc}的要求，对“{topic}”进行系统梳理，包括：定义、定理（可适当选取）、证明思路（可以略写）、以及两个典型题目和解答。整体回答控制在200到300字以内。"
    ]
    tpl = random.choice(templates)
    return tpl.format(topic=topic, difficulty_desc=difficulty_desc)


def build_chat_prompt(instruction: str) -> str:
    system_prompt = (
        "你是一名擅长讲解线性代数的大学教师，"
        "需要为学生提供清晰、循序渐进、逻辑严谨的中文讲解。"
    )
    full_prompt = (
        f"[SYSTEM]\n{system_prompt}\n\n"
        f"[USER]\n{instruction}\n\n"
        f"[ASSISTANT]\n"
    )
    return full_prompt


def _generate_one_sample(
    idx: int,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    topic = random.choice(LINEAR_ALGEBRA_TOPICS)
    difficulty_tag, difficulty_desc = random.choice(DIFFICULTY_LEVELS)

    instruction = build_instruction(topic, difficulty_tag, difficulty_desc)
    prompt = build_chat_prompt(instruction)

    try:
        response = Generation.call(
            model=model_name,
            prompt=prompt,
            max_length=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    except Exception as e:
        print(f"[{idx + 1}] 调用 DashScope 出错：{e}")
        return None

    if response.status_code != HTTPStatus.OK:
        print(
            f"[{idx + 1}] 调用失败，status={response.status_code}, "
            f"code={response.code}, msg={response.message}"
        )
        return None

    answer = str(response.output).strip()

    sample = {
        "instruction": instruction,
        "input": "",
        "output": answer,
        "meta": {
            "topic": topic,
            "difficulty": difficulty_tag,
        },
    }
    return sample


def generate_sft_samples(
    model_name: str,
    num_samples: int,
    output_path: str = "sft_linear_algebra.jsonl",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_workers: int = 4, 
) -> None:

    print(f"Using DashScope model: {model_name}")
    print(f"Using {num_workers} workers for parallel generation.")

    with open(output_path, "w", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _generate_one_sample,
                    idx,
                    model_name,
                    max_new_tokens,
                    temperature,
                    top_p,
                ): idx
                for idx in range(num_samples)
            }

            finished = 0
            for future in as_completed(futures):
                idx = futures[future]
                sample = future.result()
                if sample is None:
                    continue

                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                finished += 1

                if finished == 1 or finished % 10 == 0:
                    print(f"[{finished}/{num_samples}] Generated one sample on topic: {sample['meta']['topic']} ({sample['meta']['difficulty']})")

    print(f"\nDone! Saved {finished} samples to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sft_linear_algebra.jsonl",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_sft_samples(
        model_name=args.model,
        num_samples=args.num_samples,
        output_path=args.output,
        num_workers=args.num_workers,
    )
