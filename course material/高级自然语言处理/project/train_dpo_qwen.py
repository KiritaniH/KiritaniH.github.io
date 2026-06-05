import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Could not estimate the number of tokens")

class PrefJsonlDataset(Dataset):
    def __init__(self, path: str, max_samples: int | None = None):
        self.samples = []
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
                rc = obj.get("response_chosen", "").strip()
                rr = obj.get("response_rejected", "").strip()
                if not ins or not rc or not rr:
                    continue
                self.samples.append(
                    {
                        "instruction": ins,
                        "response_chosen": rc,
                        "response_rejected": rr,
                    }
                )

        print(f"Loaded {len(self.samples)} preference samples from {path}")
        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
            print(f"Truncated to first {max_samples} samples for faster DPO.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item
@dataclass
class PrefCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "instruction": [f["instruction"] for f in features],
            "response_chosen": [f["response_chosen"] for f in features],
            "response_rejected": [f["response_rejected"] for f in features],
        }

class SimpleDPOTrainer(Trainer):
    def __init__(
        self,
        ref_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        beta: float = 0.1,
        max_length: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self._tokenizer = tokenizer
        self.beta = beta
        self.max_length = max_length

        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()

    def _build_batch(self, prompts: List[str], answers: List[str]):
        input_ids_batch = []
        attention_masks = []
        answer_masks = []

        pad_id = self._tokenizer.pad_token_id
        eos_id = self._tokenizer.eos_token_id

        for p, a in zip(prompts, answers):
            p_ids = self._tokenizer(
                p, add_special_tokens=False
            )["input_ids"]
            a_ids = self._tokenizer(
                a, add_special_tokens=False
            )["input_ids"]
            if eos_id is not None:
                a_ids = a_ids + [eos_id]

            full_ids = p_ids + a_ids
            if len(full_ids) > self.max_length:
                overflow = len(full_ids) - self.max_length
                cut_p = min(overflow, len(p_ids))
                p_ids = p_ids[cut_p:]
                full_ids = p_ids + a_ids
            answer_start = len(full_ids) - len(a_ids)
            seq_len = len(full_ids)

            input_ids_batch.append(full_ids)
            attention_masks.append([1] * seq_len)
            ans_mask = [0] * seq_len
            for i in range(answer_start, seq_len):
                ans_mask[i] = 1
            answer_masks.append(ans_mask)

        max_len = max(len(ids) for ids in input_ids_batch)
        for i in range(len(input_ids_batch)):
            pad_len = max_len - len(input_ids_batch[i])
            input_ids_batch[i] = input_ids_batch[i] + [pad_id] * pad_len
            attention_masks[i] = attention_masks[i] + [0] * pad_len
            answer_masks[i] = answer_masks[i] + [0] * pad_len

        input_ids = torch.tensor(input_ids_batch, dtype=torch.long, device=self.model.device)
        attn_mask = torch.tensor(attention_masks, dtype=torch.long, device=self.model.device)
        ans_mask = torch.tensor(answer_masks, dtype=torch.float32, device=self.model.device)

        return input_ids, attn_mask, ans_mask

    def _compute_logps(self, model, prompts: List[str], answers: List[str]) -> torch.Tensor:
        input_ids, attn_mask, ans_mask = self._build_batch(prompts, answers)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
        )
        logits = outputs.logits 

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=input_ids.unsqueeze(-1)
        ).squeeze(-1)
        masked_log_probs = token_log_probs * ans_mask 
        seq_logps = masked_log_probs.sum(dim=-1) 
        return seq_logps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None,):
        prompts = []
        chosen_list = []
        rejected_list = []

        system_prompt = (
            "你是一名擅长讲解大学线性代数和相关数学课程的助教，"
            "需要根据学生的问题给出清晰、循序渐进、逻辑严谨的中文讲解。"
        )

        for ins, rc, rr in zip(
            inputs["instruction"],
            inputs["response_chosen"],
            inputs["response_rejected"],
        ):
            prompt = (
                f"[SYSTEM]\n{system_prompt}\n\n"
                f"[USER]\n{ins}\n\n"
                f"[ASSISTANT]\n"
            )
            prompts.append(prompt)
            chosen_list.append(rc)
            rejected_list.append(rr)

        policy_logps_chosen = self._compute_logps(model, prompts, chosen_list)
        policy_logps_rejected = self._compute_logps(model, prompts, rejected_list)

        with torch.no_grad():
            ref_logps_chosen = self._compute_logps(self.ref_model, prompts, chosen_list)
            ref_logps_rejected = self._compute_logps(self.ref_model, prompts, rejected_list)

        pi_diff = policy_logps_chosen - policy_logps_rejected
        ref_diff = ref_logps_chosen - ref_logps_rejected

        dpo_logits = self.beta * (pi_diff - ref_diff)
        loss = -F.logsigmoid(dpo_logits).mean()

        if return_outputs:
            return loss, {
                "policy_logps_chosen": policy_logps_chosen.detach(),
                "policy_logps_rejected": policy_logps_rejected.detach(),
            }
        return loss

def train_dpo(
    sft_model_dir: str,
    pref_file: str,
    output_dir: str,
    num_train_epochs: int = 1,
    batch_size: int = 1,
    grad_accum_steps: int = 8,
    lr: float = 5e-6,
    beta: float = 0.1,
    max_length: int = 512,
    max_samples: int | None = None,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        sft_model_dir,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy_model = AutoModelForCausalLM.from_pretrained(
        sft_model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32, 
    ).to(device)

    ref_model = AutoModelForCausalLM.from_pretrained(
        sft_model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)
    ref_model.eval()

    dataset = PrefJsonlDataset(pref_file)
    data_collator = PrefCollator()

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
        remove_unused_columns=False
    )

    trainer = SimpleDPOTrainer(
        model=policy_model,
        args=training_args,
        ref_model=ref_model,
        tokenizer=tokenizer,
        beta=beta,
        max_length=max_length,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"DPO training finished. Model saved to {output_dir}")
    dataset = PrefJsonlDataset(pref_file, max_samples=max_samples)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sft_model_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pref_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen_dpo_ckpt",
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
        default=5e-6,
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=384,
    )
    parser.add_argument(
    "--max_samples",
    type=int,
    default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dpo(
        sft_model_dir=args.sft_model_dir,
        pref_file=args.pref_file,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )
