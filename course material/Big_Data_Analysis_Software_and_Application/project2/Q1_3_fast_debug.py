import os
import json
import math
import argparse
import time
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model


class FastChatDataset(Dataset):
    """
    极简 SFT Dataset。

    为了先跑通 CPU debug：
    1. 只使用每条 trajectory 的最后一个 assistant 作为监督目标；
    2. system/user/interpreter/之前 assistant 作为上下文；
    3. labels 中只有最后 assistant 的 token 参与 loss，其余为 -100；
    4. 初始化阶段预 tokenize，并显示进度条。
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        max_samples: int = 8,
        max_length: int = 512,
        split_name: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = []

        print(f"[Dataset] Loading {split_name} data from {path}", flush=True)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if max_samples is not None and max_samples > 0:
            data = data[:max_samples]

        print(f"[Dataset] {split_name} raw samples used: {len(data)}", flush=True)

        for sample in tqdm(data, desc=f"Tokenizing {split_name}"):
            messages = sample["messages"]
            feat = self.encode_one(messages)
            if feat is not None:
                self.features.append(feat)

        print(f"[Dataset] {split_name} valid features: {len(self.features)}", flush=True)

        if len(self.features) == 0:
            raise ValueError(f"No valid features for {split_name}.")

    def render(self, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            text = ""
            for m in messages:
                text += f"{m['role']}:\n{m['content']}\n\n"
            if add_generation_prompt:
                text += "assistant:\n"
            return text

    def encode_one(self, messages: List[Dict[str, str]]):
        # 找最后一个 assistant 位置
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            return None

        prefix_messages = messages[:last_assistant_idx]
        answer_message = messages[last_assistant_idx]
        full_messages = messages[: last_assistant_idx + 1]

        prefix_text = self.render(prefix_messages, add_generation_prompt=True)
        full_text = self.render(full_messages, add_generation_prompt=False)

        prefix_ids = self.tokenizer(
            prefix_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        if len(full_ids) <= 2:
            return None

        labels = [-100] * len(full_ids)

        start = min(len(prefix_ids), len(full_ids))

        # 如果 prefix 已经被截断到和 full 一样长，则退化为最后 1/3 tokens 做监督
        if start >= len(full_ids):
            start = int(len(full_ids) * 0.67)

        for pos in range(start, len(full_ids)):
            labels[pos] = full_ids[pos]

        # 如果没有任何有效 label，跳过
        if all(x == -100 for x in labels):
            return None

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def collate_fn(batch, pad_token_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for x in batch:
        pad_len = max_len - len(x["input_ids"])

        input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out.loss.item()
            steps += 1

    model.train()
    avg_loss = total_loss / max(steps, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="./model")
    parser.add_argument("--train_path", type=str, default="./output/train_qwen.json")
    parser.add_argument("--val_path", type=str, default="./output/val_qwen.json")
    parser.add_argument("--output_dir", type=str, default="./qwen_lora_fast_debug")

    parser.add_argument("--max_train_samples", type=int, default=2)
    parser.add_argument("--max_val_samples", type=int, default=1)

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--torch_num_threads", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=1)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.set_num_threads(args.torch_num_threads)

    print("=" * 80, flush=True)
    print("[Step 1] Environment", flush=True)
    print("torch:", torch.__version__, flush=True)
    print("cuda available:", torch.cuda.is_available(), flush=True)
    print("torch num threads:", torch.get_num_threads(), flush=True)
    print("=" * 80, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[Step 2] Loading tokenizer...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[Step 2] Tokenizer loaded in {time.time() - t0:.2f}s", flush=True)

    print("[Step 3] Building tokenized datasets...", flush=True)
    train_dataset = FastChatDataset(
        args.train_path,
        tokenizer,
        max_samples=args.max_train_samples,
        max_length=args.max_length,
        split_name="train",
    )
    val_dataset = FastChatDataset(
        args.val_path,
        tokenizer,
        max_samples=args.max_val_samples,
        max_length=args.max_length,
        split_name="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    print("[Step 4] Loading base model...", flush=True)
    print("This may take several minutes on CPU.", flush=True)
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    print(f"[Step 4] Base model loaded in {time.time() - t0:.2f}s", flush=True)

    model.config.use_cache = False

    print("[Step 5] Adding LoRA adapters...", flush=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    print("[Step 6] Start training", flush=True)
    model.train()

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss / args.grad_accum_steps
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                pbar.set_postfix({
                    "loss": f"{loss.item() * args.grad_accum_steps:.4f}",
                    "global_step": global_step,
                })

                if global_step % args.eval_steps == 0:
                    val_loss, val_ppl = evaluate(model, val_loader, device)
                    print(
                        f"[Eval] step={global_step}, val_loss={val_loss:.4f}, val_ppl={val_ppl:.4f}",
                        flush=True,
                    )

                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint_step_{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"[Save] checkpoint saved to {ckpt_dir}", flush=True)

    final_dir = os.path.join(args.output_dir, "final_lora")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("=" * 80, flush=True)
    print(f"[Done] final LoRA saved to {final_dir}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()