import argparse
import jsonlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import util
import re
from tqdm import tqdm
import sys

MAX_INT = sys.maxsize
invalid_outputs = []

FEW_SHOT_PROMPT = r"""
### Example 1
Problem: Find the value of the expression $ 3+\sqrt{16} $.
Solution:
We compute $\\sqrt{16}=4$, then the expression is $3+4=7$.
The final answer is \\boxed{7}.

### Example 2
Problem: What is the value of $1 + 2^3$?
Solution:
We compute $2^3 = 8$, then $1+8 = 9$.
The final answer is \\boxed{9}.

### Example 3
Problem: Simplify the expression $5 - 12$.
Solution:
Compute $5 - 12 = -7$.
The final answer is \\boxed{-7}.

### Example 4
Problem: Compute $\\frac{1}{2} + \\frac{1}{3}$.
Solution:
Common denominator is 6. So $1/2 = 3/6$ and $1/3 = 2/6$.
Sum is $3/6 + 2/6 = 5/6$.
The final answer is \\boxed{\\frac{5}{6}}.

### Example 5
Problem: What is the value of $7^2$?
Solution:
We know $7^2 = 49$.
The final answer is \\boxed{49}.
"""

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s.startswith(left)
        assert s.endswith("}")
        return s[len(left):-1]
    except:
        return None

def extract_pred_boxed(text):
    # Match \boxed{...} including nested braces
    pattern = r"\\boxed\{([^{}]*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return f"\\boxed{{{matches[-1]}}}"
    return None


def process_results(question, completion, gold):
    pred_boxed = extract_pred_boxed(completion)

    if pred_boxed is None:
        invalid_outputs.append({
            "question": question,
            "output": completion,
            "gold": gold
        })
        return False

    pred = remove_boxed(pred_boxed)
    if pred is None:
        invalid_outputs.append({
            "question": question,
            "output": completion,
            "gold": gold
        })
        return False

    try:
        return util.is_equiv(pred, gold)
    except:
        return False

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    output = []
    for i in range(n - 1):
        output.append(data_list[i * batch_size:(i + 1) * batch_size])
    output.append(data_list[(n - 1) * batch_size:])
    return output


def test_hendrycks_math(model_path, data_path, k_shot=0,
                        start=0, end=MAX_INT, batch_size=1):

    zero_shot_prompt = r"""
        Solve the following math problem carefully.
        Show your reasoning step by step.
        At the end, give ONLY the final answer in LaTeX boxed form like: \\boxed{{42}}.
        Do NOT include units or explanations after the box.
        Please answer in English.
        ### Problem:\n{instruction}\n\n### Solution:\nLet's think step by step.
    """

    print("Using prompt:\n", zero_shot_prompt)

    math_in = []
    math_gold = []

    with open(data_path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            base_prompt = zero_shot_prompt.format(instruction=item["instruction"])

            if k_shot == 0:
                prompt = base_prompt
            else:
                prompt = FEW_SHOT_PROMPT + "\n\n" + base_prompt

            math_in.append(prompt)

            gold_boxed = util.last_boxed_only_string(item["output"])
            math_gold.append(remove_boxed(gold_boxed))

    math_in = math_in[start:end]
    math_gold = math_gold[start:end]

    print(f"Loaded {len(math_in)} problems.")

    batches = batch_data(math_in, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    completions = []

    for batch in tqdm(batches, desc="Running batches"):
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120, 
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id
            )

        for out in outputs:
            text = tokenizer.decode(out, skip_special_tokens=True)
            completions.append(text)

    results = []
    for q, pred, gold in zip(math_in, completions, math_gold):
        results.append(process_results(q, pred, gold))

    acc = sum(results) / len(results)
    print("\nAccuracy =", acc)
    print("Invalid:", len(invalid_outputs))

    if invalid_outputs:
        print("Example invalid output:")
        print(invalid_outputs[0])

    return acc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument("--start", type=int, default=6)
    parser.add_argument("--end", type=int, default=76)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_hendrycks_math(
        model_path=args.model,
        data_path=args.data_file,
        k_shot=args.k_shot,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size
    )
