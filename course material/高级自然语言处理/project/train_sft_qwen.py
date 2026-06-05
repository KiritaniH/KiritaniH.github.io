import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)


class SFTJsonlDataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_length: int = 1024):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

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
                self.samples.append({"instruction": ins, "output": out})

        print(f"Loaded {len(self.samples)} SFT samples from {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        instruction = item["instruction"]
        output = item["output"]

        system_prompt = (
            "你是一名擅长讲解大学线性代数和相关数学课程的助教，"
            "需要根据学生的问题给出清晰、循序渐进、逻辑严谨的中文讲解。"
        )
        text = (
            f"[SYSTEM]\n{system_prompt}\n\n"
            f"[USER]\n{instruction}\n\n"
            f"[ASSISTANT]\n{output}"
        )

        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        labels = input_ids.copy()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            padding=True,
            return_tensors="pt",
        )
        labels = [f["labels"] for f in features]
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for l in labels:
            if len(l) < max_len:
                padded_labels.append(l + [-100] * (max_len - len(l)))
            else:
                padded_labels.append(l[:max_len])
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def train_sft(
    model_name: str,
    train_file: str,
    output_dir: str,
    num_train_epochs: int = 1,
    batch_size: int = 1,
    grad_accum_steps: int = 8,
    lr: float = 5e-5,
    max_length: int = 1024,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.to(device)

    dataset = SFTJsonlDataset(train_file, tokenizer, max_length=max_length)

    n_total = len(dataset)
    n_train = int(n_total * 0.95)
    train_dataset = torch.utils.data.Subset(dataset, list(range(n_train)))
    eval_dataset = torch.utils.data.Subset(dataset, list(range(n_train, n_total)))

    data_collator = DataCollatorForCausalLM(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        warmup_ratio=0.03,
        logging_steps=10,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        report_to=[],
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training finished. Model saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen_sft_ckpt",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_sft(
        model_name=args.model,
        train_file=args.train_file,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        max_length=args.max_length,
    )
