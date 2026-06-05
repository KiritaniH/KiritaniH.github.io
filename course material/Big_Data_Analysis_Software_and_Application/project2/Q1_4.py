import os
import sqlite3
import argparse
from typing import Optional

import pandas as pd
import torch
import gradio as gr

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


SYSTEM_PROMPT = """You are a data analysis agent.

Your task is to help users solve data analysis, statistics, SQL, and mathematical modeling problems.

When a data file summary is provided:
1. First inspect the available columns, data types, sample rows, or database schema.
2. Then propose a rigorous analysis plan.
3. If code is needed, write clear Python or SQL code.
4. If the question can be answered directly from the provided information, give a concise final answer.
5. Keep the answer structured and directly related to the user question.
"""


def summarize_csv_or_excel(file_path: str, max_rows: int = 5) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        return ""

    lines = []
    lines.append(f"File path: {file_path}")
    lines.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    lines.append("")
    lines.append("Columns and dtypes:")
    for col, dtype in df.dtypes.items():
        lines.append(f"- {col}: {dtype}")

    lines.append("")
    lines.append(f"First {max_rows} rows:")
    lines.append(df.head(max_rows).to_markdown(index=False))

    missing = df.isna().sum()
    if missing.sum() > 0:
        lines.append("")
        lines.append("Missing values:")
        for col, cnt in missing.items():
            if cnt > 0:
                lines.append(f"- {col}: {int(cnt)}")

    return "\n".join(lines)


def summarize_sqlite(file_path: str, max_rows: int = 3) -> str:
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [x[0] for x in cursor.fetchall()]

    lines = []
    lines.append(f"SQLite database path: {file_path}")
    lines.append("")
    lines.append("Tables:")
    for t in tables:
        lines.append(f"- {t}")

    for table in tables:
        lines.append("")
        lines.append(f"Schema for table `{table}`:")

        cursor.execute(f"PRAGMA table_info('{table}')")
        cols = cursor.fetchall()
        for col in cols:
            # cid, name, type, notnull, dflt_value, pk
            lines.append(f"- {col[1]}: {col[2]}")

        try:
            df = pd.read_sql_query(f"SELECT * FROM '{table}' LIMIT {max_rows}", conn)
            lines.append(f"First {max_rows} rows from `{table}`:")
            lines.append(df.to_markdown(index=False))
        except Exception as e:
            lines.append(f"Could not preview table `{table}`: {e}")

    conn.close()
    return "\n".join(lines)


def summarize_uploaded_file(file_obj) -> str:
    if file_obj is None:
        return "No data file was uploaded."

    file_path = file_obj.name
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext in [".csv", ".xlsx", ".xls"]:
            return summarize_csv_or_excel(file_path)
        if ext in [".sqlite", ".db", ".sqlite3"]:
            return summarize_sqlite(file_path)
        return f"Uploaded file path: {file_path}\nUnsupported file type for automatic preview."
    except Exception as e:
        return f"Failed to summarize uploaded file: {e}"


class DataAgentDemo:
    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.lora_path = lora_path

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float32
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        if lora_path is not None and os.path.exists(lora_path):
            print(f"Loading LoRA adapter from: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        else:
            print("No LoRA adapter loaded. Using base model only.")

        self.model.to(self.device)
        self.model.eval()

    def build_prompt(self, user_question: str, file_summary: str) -> list:
        content = f"""User question:
{user_question}

Data file summary:
{file_summary}

Please solve the task as a data analysis agent.
"""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def generate(
        self,
        user_question: str,
        file_obj,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        if not user_question or not user_question.strip():
            return "Please enter a question."

        file_summary = summarize_uploaded_file(file_obj)
        messages = self.build_prompt(user_question.strip(), file_summary)

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        do_sample = temperature > 0

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)

        return answer.strip()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="./model")
    parser.add_argument("--lora_path", type=str, default="./qwen_lora_ray_output/final_lora")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--base_only", action="store_true")

    args = parser.parse_args()

    lora_path = None if args.base_only else args.lora_path

    agent = DataAgentDemo(
        model_path=args.model_path,
        lora_path=lora_path,
    )

    with gr.Blocks(title="Data Agent Demo") as demo:
        gr.Markdown(
            """
# Data Agent Demo

This demo uses a Qwen-based data analysis agent.  
You can ask data analysis, statistics, SQL, machine learning, or mathematical modeling questions.  
You may also upload a CSV, Excel, or SQLite file. The system will automatically extract a small schema/preview and send it to the model.
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload data file: CSV / Excel / SQLite",
                    file_types=[".csv", ".xlsx", ".xls", ".sqlite", ".db", ".sqlite3"],
                )

                question = gr.Textbox(
                    label="Question",
                    lines=6,
                    placeholder="Example: Please inspect this dataset and suggest a classification pipeline.",
                )

                max_new_tokens = gr.Slider(
                    minimum=64,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="Max new tokens",
                )

                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.2,
                    step=0.05,
                    label="Temperature",
                )

                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-p",
                )

                run_button = gr.Button("Run Data Agent")

            with gr.Column(scale=1):
                output = gr.Textbox(
                    label="Agent output",
                    lines=24,
                )

        run_button.click(
            fn=agent.generate,
            inputs=[question, file_input, max_new_tokens, temperature, top_p],
            outputs=output,
        )

        gr.Examples(
            examples=[
                [
                    None,
                    "I have a dataset with customer features and a binary purchase label. Please design a reasonable machine learning pipeline.",
                    512,
                    0.2,
                    0.9,
                ],
                [
                    None,
                    "Explain how to analyze missing values, outliers, and feature correlations in a tabular dataset.",
                    512,
                    0.2,
                    0.9,
                ],
                [
                    None,
                    "For a SQLite database, how should I inspect the schema and write a query to aggregate records by category?",
                    512,
                    0.2,
                    0.9,
                ],
            ],
            inputs=[file_input, question, max_new_tokens, temperature, top_p],
        )

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()