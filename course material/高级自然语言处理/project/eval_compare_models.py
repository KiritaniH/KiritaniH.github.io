import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_BASE = "Qwen/Qwen2.5-0.5B"
MODEL_SFT = r"C:\Users\lenovo\Desktop\advnlp\project\qwen_sft_ckpt"
MODEL_DPO = r"C:\Users\lenovo\Desktop\advnlp\project\qwen_dpo_ckpt"
JUDGE_MODEL = MODEL_BASE
TEST_QUESTIONS = [
    "请解释特征值与特征向量的意义，并举例说明。",
    "判断矩阵 [[1,2],[0,1]] 是否可对角化，并说明理由。",
    "什么是秩？秩与列空间、零空间之间有什么关系？",
    "如何判断一组向量是否线性无关？请给出步骤。",
    "请解释矩阵的逆的定义及意义。",
]

SYSTEM_PROMPT = (
    "你是一名擅长讲解大学线性代数的助教，需要给出清晰、循序渐进的中文讲解。"
)


def load_model(path: str):
    print(f"Loading model from: {path}")
    tok = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=not ("/" in path and ":" not in path),
    )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=not ("/" in path and ":" not in path),
        dtype=torch.float32,
    ).to(device)

    return tok, model


def build_chat_prompt(question: str) -> str:
    return (
        f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n"
        f"[USER]\n{question}\n\n"
        f"[ASSISTANT]\n"
    )


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    prompt = build_chat_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[ASSISTANT]" in text:
        text = text.split("[ASSISTANT]", 1)[-1].strip()
    return text


def judge_pair(
    judge_model,
    judge_tokenizer,
    question: str,
    ans_a: str,
    ans_b: str,
    label_a: str,
    label_b: str,
    max_new_tokens: int = 64,
) -> str:
    judge_prompt = f"""
你是一个负责打分的评估助手，现在需要根据“正确性、逻辑性、严谨性和解释清晰度”比较两个回答的优劣。

【问题】：
{question}

【回答A】（模型 {label_a}）：
{ans_a}

【回答B】（模型 {label_b}）：
{ans_b}

请根据线性代数的知识，判断哪个回答整体更好。

要求：
1. 只考虑数学正确性、推理合理性、解释是否清晰。
2. 不用给出分析过程。
3. 最终输出严格限制为一个大写字母：A 或 B。

现在请给出你的选择：
""".strip()

    inputs = judge_tokenizer(judge_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    text = judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
    text_up = text.upper()
    if "A" in text_up and "B" in text_up:
        last_line = text_up.strip().split("\n")[-1]
        if "A" in last_line and "B" not in last_line:
            return "A"
        if "B" in last_line and "A" not in last_line:
            return "B"
    if "A" in text_up and "B" not in text_up:
        return "A"
    if "B" in text_up and "A" not in text_up:
        return "B"
    return "unknown"


def main():
    print("Loading models...")
    tok_base, model_base = load_model(MODEL_BASE)
    tok_sft, model_sft = load_model(MODEL_SFT)
    tok_dpo, model_dpo = load_model(MODEL_DPO)

    tok_judge, model_judge = load_model(JUDGE_MODEL)

    results = []

    print("Running evaluation...")
    for q in TEST_QUESTIONS:
        print("\n========== Question ==========")
        print(q)

        ans_base = generate_answer(model_base, tok_base, q)
        ans_sft = generate_answer(model_sft, tok_sft, q)
        ans_dpo = generate_answer(model_dpo, tok_dpo, q)

        print("\n--- Base ---\n", ans_base)
        print("\n--- SFT ---\n", ans_sft)
        print("\n--- DPO ---\n", ans_dpo)

        # LLM-as-judge：Base vs SFT, SFT vs DPO
        judge_base_vs_sft = judge_pair(
            model_judge,
            tok_judge,
            q,
            ans_base,
            ans_sft,
            "Base",
            "SFT",
        )

        judge_sft_vs_dpo = judge_pair(
            model_judge,
            tok_judge,
            q,
            ans_sft,
            ans_dpo,
            "SFT",
            "DPO",
        )

        print(f"\n[Judge] Base vs SFT: {judge_base_vs_sft}")
        print(f"[Judge] SFT vs DPO: {judge_sft_vs_dpo}")

        results.append(
            {
                "question": q,
                "base": ans_base,
                "sft": ans_sft,
                "dpo": ans_dpo,
                "judge_base_vs_sft": judge_base_vs_sft,
                "judge_sft_vs_dpo": judge_sft_vs_dpo,
            }
        )

    out_path = os.path.join(os.path.dirname(__file__), "eval_results_with_judge.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nAll results (with judge) saved to {out_path}")


if __name__ == "__main__":
    main()
