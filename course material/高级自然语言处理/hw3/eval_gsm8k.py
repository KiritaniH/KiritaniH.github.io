import argparse
import json
import re
import jsonlines
from fraction import Fraction
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
MAX_INT = sys.maxsize

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) and is_number(numerator):
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        return round(float(frac.numerator / frac.denominator))
                return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
    return None


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    output = []
    for i in range(n - 1):
        output.append(data_list[i * batch_size: (i + 1) * batch_size])
    output.append(data_list[(n - 1) * batch_size:])
    return output


FEW_SHOT_COT_PROMPT = """Question: John has 3 apples. He buys 5 more apples. How many apples does he have now?
Let's think step by step.
First John has 3 apples.
He buys 5 more, so we add 3 + 5 = 8.
The answer is: 8

Question: A box contains 6 red balls and 4 blue balls. How many balls are there in total?
Let's think step by step.
There are 6 red balls and 4 blue balls.
Total = 6 + 4 = 10.
The answer is: 10

Question: Sarah reads 12 pages on Monday and 15 pages on Tuesday. How many pages does she read in total?
Let's think step by step.
On Monday she reads 12 pages.
On Tuesday she reads 15 pages.
Total pages = 12 + 15 = 27.
The answer is: 27

Question: A car travels 40 km in the morning and 35 km in the afternoon. How many kilometers in total?
Let's think step by step.
Total = 40 + 35 = 75.
The answer is: 75

Question: There are 5 bags, each with 4 candies. How many candies?
Let's think step by step.
Total = 5 × 4 = 20.
The answer is: 20
"""

def build_prompt(query, k_shot):
    zero_shot_template = (
        "Below is a math problem.\n\n"
        "### Problem:\n{instruction}\n\n"
        "### Solution: Let's think step by step.\n"
        "Please give the answer in the form: The answer is: <number>."
    )

    if k_shot == 0:
        return zero_shot_template.format(instruction=query)
    else:
        return FEW_SHOT_COT_PROMPT + "\n\n" + f"""
        Question: {query}
        Let's think step by step.
        Finally, give the answer in the form: The answer is: <number>.
        Please answer in English.
        """


def gsm8k_test(model_path, data_path, k_shot, start=0, end=MAX_INT, batch_size=1):
    print(f"Using k_shot = {k_shot}")

    gsm8k_ins = []
    gsm8k_answers = []

    with open(data_path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            prompt = build_prompt(item["query"], k_shot)
            gsm8k_ins.append(prompt)

            gold = int(item["response"].split("#### ")[1].replace(",", ""))
            gsm8k_answers.append(gold)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    model.eval()

    batches = batch_data(gsm8k_ins, batch_size=batch_size)
    completions = []

    for batch in batches:
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
            )

        for out in outputs:
            completions.append(tokenizer.decode(out, skip_special_tokens=True))
    results = []
    for gold, pred in zip(gsm8k_answers, completions):
        extracted = extract_answer_number(pred)
        results.append(extracted == gold)

    acc = sum(results) / len(results)
    print("Accuracy:", acc)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gsm8k_test(
        model_path=args.model,
        data_path=args.data_file,
        k_shot=args.k_shot,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size
    )
