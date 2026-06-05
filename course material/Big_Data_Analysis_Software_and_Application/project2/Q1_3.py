import os

os.environ["RAY_DISABLE_DASHBOARD"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

import json
import math
import time
import argparse
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

import ray
import ray.train as train
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model


# ============================================================
# Dataset
# ============================================================
class FastChatDataset(Dataset):
    """
    CPU-friendly SFT dataset.

    For each trajectory:
    1. Keep the original multi-turn messages.
    2. Use the final assistant response as the supervised target.
    3. Use previous messages as context.
    4. Mask non-target tokens with -100.
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

        print(f"[Dataset] {split_name} samples used: {len(data)}", flush=True)

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
        last_assistant_idx = None

        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            return None

        prefix_messages = messages[:last_assistant_idx]
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

        # If the prefix itself is too long and gets truncated,
        # use the last 1/3 tokens as the supervised assistant target.
        if start >= len(full_ids):
            start = int(len(full_ids) * 0.67)

        for pos in range(start, len(full_ids)):
            labels[pos] = full_ids[pos]

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


# ============================================================
# Eval
# ============================================================
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


# ============================================================
# Ray Train worker function
# ============================================================
def train_loop_per_worker(config):
    print("=" * 80, flush=True)
    print("[Ray Worker] Training started", flush=True)
    print("=" * 80, flush=True)

    torch.set_num_threads(config["torch_num_threads"])

    print("[Step 1] Environment", flush=True)
    print("torch:", torch.__version__, flush=True)
    print("cuda available:", torch.cuda.is_available(), flush=True)
    print("torch num threads:", torch.get_num_threads(), flush=True)

    use_gpu = config["use_gpu"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print("device:", device, flush=True)

    model_path = config["model_path"]
    train_path = config["train_path"]
    val_path = config["val_path"]
    output_dir = config["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------
    # Tokenizer
    # ----------------------------
    print("[Step 2] Loading tokenizer...", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Step 2] Tokenizer loaded in {time.time() - t0:.2f}s", flush=True)

    # ----------------------------
    # Dataset
    # ----------------------------
    print("[Step 3] Building datasets...", flush=True)

    train_dataset = FastChatDataset(
        train_path,
        tokenizer,
        max_samples=config["max_train_samples"],
        max_length=config["max_length"],
        split_name="train",
    )

    val_dataset = FastChatDataset(
        val_path,
        tokenizer,
        max_samples=config["max_val_samples"],
        max_length=config["max_length"],
        split_name="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    # ----------------------------
    # Model
    # ----------------------------
    print("[Step 4] Loading base model...", flush=True)
    print("This may take several minutes on CPU.", flush=True)
    t0 = time.time()

    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    print(f"[Step 4] Base model loaded in {time.time() - t0:.2f}s", flush=True)

    model.config.use_cache = False

    # ----------------------------
    # LoRA
    # ----------------------------
    print("[Step 5] Adding LoRA adapters...", flush=True)

    target_modules = config["target_modules"].split(",")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"])

    # ----------------------------
    # Training
    # ----------------------------
    print("[Step 6] Start training", flush=True)

    model.train()

    global_step = 0
    best_val_loss = float("inf")

    optimizer.zero_grad()

    for epoch in range(config["epochs"]):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}")

        running_loss = 0.0

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss / config["grad_accum_steps"]
            loss.backward()

            running_loss += loss.item() * config["grad_accum_steps"]

            if (step + 1) % config["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                avg_train_loss = running_loss / max(step + 1, 1)

                pbar.set_postfix({
                    "loss": f"{avg_train_loss:.4f}",
                    "global_step": global_step,
                })

                if global_step % config["eval_steps"] == 0:
                    val_loss, val_ppl = evaluate(model, val_loader, device)

                    print(
                        f"[Eval] step={global_step}, "
                        f"train_loss={avg_train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, "
                        f"val_ppl={val_ppl:.4f}",
                        flush=True,
                    )

                    train.report({
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                    })

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                        best_dir = os.path.join(output_dir, "best_lora")
                        os.makedirs(best_dir, exist_ok=True)

                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)

                        print(f"[Save] best checkpoint saved to {best_dir}", flush=True)

                if global_step % config["save_steps"] == 0:
                    ckpt_dir = os.path.join(output_dir, f"checkpoint_step_{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)

                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

                    print(f"[Save] checkpoint saved to {ckpt_dir}", flush=True)

    # ----------------------------
    # Final save
    # ----------------------------
    final_dir = os.path.join(output_dir, "final_lora")
    os.makedirs(final_dir, exist_ok=True)

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("=" * 80, flush=True)
    print(f"[Done] final LoRA saved to {final_dir}", flush=True)
    print("=" * 80, flush=True)


# ============================================================
# Main
# ============================================================

@ray.remote
def ray_ping():
    import os
    return {
        "pid": os.getpid(),
        "cwd": os.getcwd(),
    }

    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="./model")
    parser.add_argument("--train_path", type=str, default="./output/train_qwen.json")
    parser.add_argument("--val_path", type=str, default="./output/val_qwen.json")
    parser.add_argument("--output_dir", type=str, default="./qwen_lora_ray_output")

    parser.add_argument("--max_train_samples", type=int, default=8)
    parser.add_argument("--max_val_samples", type=int, default=4)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated LoRA target modules. CPU debug default: q_proj,v_proj",
    )

    parser.add_argument("--torch_num_threads", type=int, default=4)

    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=2)

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--use_gpu", action="store_true")

    args = parser.parse_args()

    # Convert to absolute paths because Ray workers may have different working directory.
    args.model_path = os.path.abspath(args.model_path)
    args.train_path = os.path.abspath(args.train_path)
    args.val_path = os.path.abspath(args.val_path)
    args.output_dir = os.path.abspath(args.output_dir)

    config = vars(args)

    print("=" * 80, flush=True)
    print("[Main] Ray Train Qwen LoRA SFT", flush=True)
    print("model_path:", args.model_path, flush=True)
    print("train_path:", args.train_path, flush=True)
    print("val_path:", args.val_path, flush=True)
    print("output_dir:", args.output_dir, flush=True)
    print("=" * 80, flush=True)

    # Avoid dashboard port problem.
    print("[Main] Starting Ray in local_mode...", flush=True)
    
    ray.shutdown()
    ray.init(
        address="local",
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=True,
        num_cpus=2,
        _temp_dir="/tmp/ray_project2",
    )

    print("[Main] Ray initialized.", flush=True)
    print("[Main] Ray resources:", ray.available_resources(), flush=True)

    print("[Main] Testing Ray worker...", flush=True)
    ping_result = ray.get(ray_ping.remote(), timeout=30)
    print("[Main] Ray worker test passed:", ping_result, flush=True)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=config,
        scaling_config=ScalingConfig(
            num_workers=1,
            use_gpu=False,
            resources_per_worker={"CPU": 1},
        ),
        run_config=RunConfig(
            name="qwen_lora_sft_ray_local",
            storage_path=args.output_dir,
        ),
    )

    print("[Main] TorchTrainer created.", flush=True)
    print("[Main] Calling trainer.fit()...", flush=True)

    result = trainer.fit()

    print("[Main] trainer.fit() finished.", flush=True)

    print("=" * 80, flush=True)
    print("[Main] Ray Train finished", flush=True)
    print(result, flush=True)
    print("=" * 80, flush=True)

    ray.shutdown()


if __name__ == "__main__":
    main()