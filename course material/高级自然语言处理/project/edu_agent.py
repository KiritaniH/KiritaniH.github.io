import json
import random
from typing import Dict, Literal, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME =r"C:\Users\lenovo\Desktop\advnlp\project\qwen_dpo_ckpt"


device = "cuda" if torch.cuda.is_available() else "cpu"

def clean_output(text: str) -> str:
    bad_tags = ["[SYSTEM]", "[USER]", "[ASSISTANT]", "ASSIGNMENT", "ASSIGNMENTS"]
    for tag in bad_tags:
        text = text.replace(tag, "")
    return text.strip()


def load_model(model_name: str):

    local_only = ":" not in model_name and "/" in model_name 

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_only,
        torch_dtype=torch.float32,
    ).to(device)

    return tokenizer, model


def chat_llm(
    tokenizer,
    model,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 1024,
) -> str:
    full_prompt = (
        f"[SYSTEM]\n{system_prompt}\n\n"
        f"[USER]\n{user_prompt}\n\n"
        f"[ASSISTANT]\n"
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    text = clean_output(text)
    return text

SYSTEM_BASE = (
    "你是一名严谨的大学线性代数助教。"
    "你需要根据学生表现进行讲解、分析和诊断，语言清晰、条理分明，避免废话。"
)

QUIZ_SYSTEM = (
    "你是线性代数练习题生成助手。"
    "请根据要求生成题目，但尽量简短，不要自我介绍，不要问问题。"
    "允许简单过渡语，但必须给出一道题目。"
)


QUIZ_PROMPT_TEMPLATE = """
请基于以下知识点生成一道{difficulty}难度的练习题：

知识点：{kp_desc}

题目请用一句或多句中文完整描述，题干前不需要任何标记。
不要给出答案。
"""


ANSWER_PROMPT_TEMPLATE = """
你是线性代数评分系统。请为下面这道题生成简洁准确的标准答案。

题目：
{question}

要求：
1. 只输出标准答案，不要重复题目。
2. 不要解释生成过程，不要输出“解析如下”等提示语。
3. 用简洁数学语言作答，可以包含必要的计算步骤。
"""

EXPLAIN_PROMPT_TEMPLATE = """
你是线性代数讲解老师，需要针对下面的知识点给出一次性完整讲解。

知识点说明：{kp_desc}
错误类型：{error_type}

请你面向学生进行讲解，要求：
1. 先给出定义，再给直观解释，再给一个简单例子。
2. 如果错误类型是“概念性错误”，重点澄清概念和常见误解。
3. 如果错误类型是“粗心计算错误”，重点强调容易出错的步骤和检查方法。
4. 不要向学生提问，不要要求学生“再给你一个例题”，
   不要扮演学生角色，也不要说“好的，我明白了”等对话语句。
5. 使用中文分段说明即可。

现在开始你的讲解：
"""

DIAGNOSE_PROMPT_TEMPLATE = """
你是线性代数助教，负责判断学生对某道题的作答是否正确，并判断错误类型。

【题目】：
{question}

【标准答案】：
{answer}

【学生作答】：
{student_answer}

请你判断：
1. 学生的作答是否整体正确（只考虑数学意义上的正确性）。
2. 如果不正确，错误主要属于哪一类：
   - 概念性错误（对定义、定理或整体思路理解错误）
   - 粗心计算错误（思路是对的，但某一步计算或符号错误）

请用 JSON 格式输出，包含字段：
- "is_correct": true 或 false
- "error_type": "concept" 或 "careless" 或 "other"

只输出 JSON，不要输出多余文字。
"""


CHAPTERS = {
    "Chapter 1: Vectors and Matrices Basics": {
        "vector_operations": "向量的加减、数乘、范数与内积，理解向量的几何意义",
        "matrix_basics": "矩阵的加法、数乘、乘法及其维度匹配，矩阵转置与分块矩阵",
    },
    "Chapter 2: Linear Systems and Echelon Forms": {
        "gaussian_elimination": "高斯消元法与行初等变换，把矩阵化为行阶梯形/行最简形",
        "rank_and_solutions": "矩阵的秩、线性方程组解的存在性与唯一性，与秩和未知数个数的关系",
        "homogeneous_vs_inhomogeneous": "齐次与非齐次线性方程组的结构及其解空间形式",
    },
    "Chapter 3: Vector Spaces and Subspaces": {
        "span_and_basis": "生成子空间、线性无关、基与维数的概念与基本例子",
        "column_row_null_spaces": "列空间、行空间与零空间的定义和关系，理解 rank-nullity 定理的含义",
    },
    "Chapter 4: Determinants and Invertibility": {
        "det_definition_properties": "行列式的定义、基本性质、对行变换的反应",
        "invertible_matrices": "可逆矩阵的等价刻画：det≠0、秩为 n、列向量线性无关等",
        "matrix_inverse_computation": "逆矩阵的计算方法（伴随矩阵法、消元法），以及解方程组的视角",
    },
    "Chapter 5: Eigenvalues, Eigenvectors and Diagonalization": {
        "eigen_def_charpoly": "特征值与特征向量的定义、特征多项式、代数重数的含义",
        "geom_mult_diag": "几何重数与特征子空间，对角化的条件及对角化矩阵的构造",
        "similarity_transform": "相似变换的概念，矩阵与线性变换在不同基下的表示关系",
    },
    "Chapter 6: Inner Product, Orthogonality and Least Squares": {
        "inner_product_spaces": "内积、长度、夹角，正交与标准正交基，Gram–Schmidt 正交化的思想",
        "orthogonal_projection": "向子空间的正交投影，如何用投影解释最小二乘问题",
        "least_squares": "最小二乘问题、正规方程 A^T A x = A^T b 的推导和含义",
    },
    "Chapter 7: Singular Value Decomposition (SVD)": {
        "svd_definition": "奇异值分解 A = U Σ V^T 的形式及其中各个矩阵的含义",
        "svd_geometry": "从几何角度理解 SVD：伸缩 + 旋转，秩与奇异值个数的关系",
        "low_rank_approx": "基于 SVD 的低秩近似，如何用前 k 个奇异值近似原矩阵",
    },
    "Chapter 8: Pseudoinverse and Applications": {
        "moore_penrose_pinv": "Moore–Penrose 伪逆的定义与四个条件，如何直观理解伪逆",
        "svd_pinv": "用 SVD 表达伪逆 A^+ = V Σ^+ U^T 的公式及其推导思路",
        "least_squares_via_pinv": "利用伪逆求解超定/欠定系统的最小二乘解及其几何解释",
    },
}

Difficulty = Literal["easy", "medium", "hard"]
StudentState = Dict[str, Dict[str, str]] 


def init_student_state() -> StudentState:
    state: StudentState = {}
    for chapter, kps in CHAPTERS.items():
        state[chapter] = {}
        for kp in kps:
            state[chapter][kp] = "none"
    return state


def pick_unmastered_kp(state: StudentState) -> Tuple[str, str]:
    candidates = []
    for chapter, kps in state.items():
        for kp_key, level in kps.items():
            if level not in ("medium_mastered", "hard_mastered"):
                candidates.append((chapter, kp_key))
    if not candidates:
        return None, None
    return random.choice(candidates)


def generate_question(tokenizer, model, kp_desc, difficulty):
    prompt = QUIZ_PROMPT_TEMPLATE.format(kp_desc=kp_desc, difficulty=difficulty)
    raw = chat_llm(tokenizer, model, QUIZ_SYSTEM, prompt, max_new_tokens=200)

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) >= 2 and ("我是" in lines[0] or "你好" in lines[0]):
        question = lines[1]
    else:
        question = lines[0] if lines else "（生成失败）"

    return {"question": question}





def generate_standard_answer(
    tokenizer,
    model,
    question: str,
) -> str:
    user_prompt = ANSWER_PROMPT_TEMPLATE.format(question=question)
    answer = chat_llm(tokenizer, model, SYSTEM_BASE, user_prompt, max_new_tokens=1024)
    return answer


def explain_concept(
    tokenizer,
    model,
    kp_desc: str,
    error_type: str,
) -> str:
    user_prompt = EXPLAIN_PROMPT_TEMPLATE.format(
        kp_desc=kp_desc,
        error_type=error_type,
    )
    explanation = chat_llm(tokenizer, model, SYSTEM_BASE, user_prompt, max_new_tokens=1024)
    return explanation


def diagnose_answer(
    tokenizer,
    model,
    question: str,
    gold_answer: str,
    student_answer: str,
) -> Dict[str, str]:
    user_prompt = DIAGNOSE_PROMPT_TEMPLATE.format(
        question=question,
        answer=gold_answer,
        student_answer=student_answer,
    )
    raw = chat_llm(tokenizer, model, SYSTEM_BASE, user_prompt, max_new_tokens=1024)
    try:
        json_str = raw[raw.index("{"): raw.rindex("}") + 1]
        data = json.loads(json_str)
    except Exception:
        data = {"is_correct": False, "error_type": "other"}
    return data


def run_learning_session_for_kp(
    tokenizer,
    model,
    chapter: str,
    kp_key: str,
    state: StudentState,
):
    kp_desc = CHAPTERS[chapter][kp_key]
    current_level = state[chapter][kp_key]

    print(f"\n=== 本次学习知识点: {kp_desc} ({chapter}) ===")
    print(f"当前掌握状态: {current_level}")

    print("\n[系统] 你觉得自己会做这个知识点相关的题吗？")
    self_eval = input("[学生] 输入 y 表示会做，输入 n 表示不会做：").strip().lower()

    def ask_one_question(difficulty: Difficulty) -> Tuple[str, str, Dict[str, str]]:
        q_pack = generate_question(tokenizer, model, kp_desc, difficulty)
        question = q_pack["question"]
        gold_answer = generate_standard_answer(tokenizer, model, question)
        print(f"\n[{difficulty.upper()} 题] {question}")
        stu_ans = input("[学生作答] ").strip()
        diag = diagnose_answer(tokenizer, model, question, gold_answer, stu_ans)
        print("[诊断 Agent] ", diag)
        return question, gold_answer, diag

    if self_eval == "n":
        print("\n[系统] 好的，我们先从讲解开始。")
        explanation = explain_concept(tokenizer, model, kp_desc, "概念性错误")
        print("\n[讲解 Agent]\n", explanation)

        print("\n[系统] 现在来做一道简单题检验一下理解。")
        _, _, diag = ask_one_question("easy")

        if diag.get("is_correct") is True:
            print("\n[系统] 很好，你已经掌握了该知识点的基础题。")
            state[chapter][kp_key] = "easy_mastered"
        else:
            print("\n[系统] 看起来还需要多练习。你可以稍后再回来复习这个知识点。")
            state[chapter][kp_key] = "none"
            return

    else: 
        print("\n[系统] 那我们先用一题简单题来确认一下。")
        _, _, diag = ask_one_question("easy")

        if diag.get("is_correct") is True:
            print("\n[系统] 很好，你在简单题上的表现不错，我们尝试提升难度。")
            state[chapter][kp_key] = "easy_mastered"
        else:
            err_type = diag.get("error_type", "other")
            if err_type == "concept":
                print("\n[系统] 你在概念上有一些误解，我们详细讲解一下。")
                explanation = explain_concept(tokenizer, model, kp_desc, "概念性错误")
                print("\n[讲解 Agent]\n", explanation)
            elif err_type == "careless":
                print("\n[系统] 看起来更像是粗心计算错误，我们回顾一下关键步骤。")
                explanation = explain_concept(tokenizer, model, kp_desc, "粗心计算错误")
                print("\n[讲解 Agent]\n", explanation)
            else:
                print("\n[系统] 错误类型不太明确，我们再系统讲解一遍。")
                explanation = explain_concept(tokenizer, model, kp_desc, "概念性错误")
                print("\n[讲解 Agent]\n", explanation)

            print("\n[系统] 再来一道相同难度的简单题试试。")
            _, _, diag2 = ask_one_question("easy")
            if diag2.get("is_correct") is True:
                print("\n[系统] 这次做对了，基础题应该没问题了。")
                state[chapter][kp_key] = "easy_mastered"
            else:
                print("\n[系统] 看起来还需要再巩固一下，我们暂时不升级难度。")
                state[chapter][kp_key] = "none"
                return

    print("\n[系统] 接下来我们做几道中等难度的题目巩固一下。")
    correct_cnt = 0
    total = 4

    for i in range(1, total + 1):
        print(f"\n[系统] 中等难度题 {i}/{total}")
        _, _, diag_m = ask_one_question("medium")
        if diag_m.get("is_correct") is True:
            correct_cnt += 1

    print(
        f"\n[系统] 中等难度题答对 {correct_cnt} / {total}。"
        "要求至少 3 题正确才算通过。"
    )

    if correct_cnt >= 3:
        print("\n[系统] 恭喜，你在中等难度上已经掌握该知识点。")
        state[chapter][kp_key] = "medium_mastered"
    else:
        print("\n[系统] 正确率暂时不够，我们再回顾一下关键概念。")
        explanation = explain_concept(tokenizer, model, kp_desc, "概念性错误")
        print("\n[讲解 Agent]\n", explanation)
        return

    print("\n[系统] 你已经在中等难度上掌握该知识点，是否要挑战更困难的题目？")
    go_hard = input("[学生] 输入 y 表示继续，其他键跳过困难题：").strip().lower()
    if go_hard != "y":
        print("\n[系统] 好的，我们可以在之后的学习中再继续挑战更高难度。")
        return

    print("\n[系统] 那我们来尝试 2 道困难题。")
    for i in range(1, 3):
        print(f"\n[系统] 困难题 {i}/2")
        ask_one_question("hard")

    print("\n[系统] 困难题部分结束，这一轮关于该知识点的学习完成。")
    state[chapter][kp_key] = "hard_mastered"


def main():
    print("=== 教学 Copilot: 线性代数多智能体系统 ===")
    print(f"使用模型: {MODEL_NAME}")
    tokenizer, model = load_model(MODEL_NAME)
    state = init_student_state()

    chapter, kp_key = pick_unmastered_kp(state)
    if chapter is None:
        print("所有知识点都已掌握，无需学习。")
        return

    run_learning_session_for_kp(tokenizer, model, chapter, kp_key, state)

    print("\n[当前学生知识状态（该章节）]:")
    print(state[chapter])


if __name__ == "__main__":
    main()
