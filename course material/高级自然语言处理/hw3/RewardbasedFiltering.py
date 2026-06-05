import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K = 5
SEED_DATA = [
    {
        "instruction": "Explain the difference between supervised and unsupervised learning.",
        "keywords": ["supervised", "unsupervised", "labeled", "labels", "data"]
    },
    {
        "instruction": "What is overfitting in machine learning and how can we prevent it?",
        "keywords": ["overfitting", "training", "regularization", "validation", "complex"]
    },
    {
        "instruction": "Describe the main idea of gradient descent.",
        "keywords": ["gradient", "descent", "loss", "update", "parameter"]
    },
    {
        "instruction": "What is a language model in natural language processing?",
        "keywords": ["language model", "probability", "token", "sequence", "predict"]
    },
    {
        "instruction": "Explain the concept of attention in the Transformer architecture.",
        "keywords": ["attention", "query", "key", "value", "weights"]
    },
    {
        "instruction": "What is the difference between precision and recall?",
        "keywords": ["precision", "recall", "true positive", "false positive", "false negative"]
    },
    {
        "instruction": "Explain the idea of word embeddings.",
        "keywords": ["embedding", "vector", "semantics", "similarity", "representation"]
    },
    {
        "instruction": "What is reinforcement learning?",
        "keywords": ["agent", "environment", "reward", "policy", "state"]
    },
    {
        "instruction": "Describe the train-validation-test split and its purpose.",
        "keywords": ["train", "validation", "test", "split", "evaluation"]
    },
    {
        "instruction": "Explain what a confusion matrix is.",
        "keywords": ["confusion matrix", "true positive", "true negative", "false positive", "false negative"]
    },
]


def compute_length_score(text, min_len=30, max_len=200):
    n = len(text.split())
    if n <= 0:
        return 0.0

    target = (min_len + max_len) / 2
    diff = abs(n - target)
    max_diff = target - min_len
    score = max(0.0, 1.0 - diff / max_diff)
    return score


def compute_keyword_score(text, keywords):
    if not keywords:
        return 0.0

    text_lower = text.lower()
    hit = 0
    for kw in keywords:
        if kw.lower() in text_lower:
            hit += 1
    return hit / len(keywords)

def compute_completeness_score(text):
    text = text.strip()
    if len(text) < 20:
        return 0.2
    if text.endswith("."):
        return 1.0
    return 0.7

def compute_repetition_penalty(text):
    words = text.lower().split()
    n = 3
    seen = set()
    repeats = 0
    for i in range(len(words) - n):
        gram = tuple(words[i:i+n])
        if gram in seen:
            repeats += 1
        seen.add(gram)
    return max(0.0, 1 - repeats / 5)

def compute_fluency_score(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
    if avg_len < 5 or avg_len > 40:
        return 0.5
    return 1.0

def compute_reward(text, keywords, alpha=0.35):
    length_score = compute_length_score(text)
    keyword_score = compute_keyword_score(text, keywords)
    repetition_score = compute_repetition_penalty(text)
    completeness = compute_completeness_score(text)
    fluency = compute_fluency_score(text)
    return (
        0.10 * length_score +
        0.25 * keyword_score +
        0.15 * repetition_score +
        0.35 * completeness +
        0.15 * fluency
    )


def load_model(model_name=MODEL_NAME, device=DEVICE):
    print(f"Loading model {model_name} on device {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return tokenizer, model


def generate_candidates(tokenizer, model, instruction, k=K, max_new_tokens=128):
    device = next(model.parameters()).device
    prompt = (
        f"Instruction: {instruction}\n\n"
        "Response:"
    )

    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    candidates = []

    for i in range(k):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,   
                temperature=0.6,
                top_p=0.9,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Response:" in text:
            text = text.split("Response:", 1)[1].strip()
        candidates.append(text)

    return candidates

def main():
    tokenizer, model = load_model()

    for idx, item in enumerate(SEED_DATA, start=1):
        instr = item["instruction"]
        keywords = item["keywords"]

        print("=" * 80)
        print(f"[{idx}] Prompt: {instr}\n")

        candidates = generate_candidates(tokenizer, model, instr, k=K)
        scores = []
        for i, cand in enumerate(candidates, start=1):
            score = compute_reward(cand, keywords)
            scores.append(score)
            print(f"Candidate {i}:")
            print(cand)
            print(f"Score = {score:.4f}\n")
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        print(f"Selected: Candidate {best_idx + 1} (Score = {scores[best_idx]:.4f})")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
