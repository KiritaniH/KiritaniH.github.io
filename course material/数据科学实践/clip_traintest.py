import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast


@dataclass
class Config:
    project_root: str = os.path.dirname(os.path.abspath(__file__))
    dataset_root: str = os.path.join(project_root, "dataset_sam3_01")
    clip_model_dir: str = os.path.join(project_root, "clip_model")
    output_dir: str = os.path.join(project_root, "logs_clip_final")

    image_size: int = 336
    batch_size: int = 16

    target_positive_ratio: float = 0.42

    # 先用短 prompt 做稳健 baseline，避免过具体 prompt 带偏
    class0_prompts: Tuple[str, ...] = (
        "a photo of an ordinary rock",
        "a photo of a normal stone",
        "a photo of a terrestrial rock",
        "a photo of a common rock",
        #"a photo of a non-meteorite rock",
    )
    class1_prompts: Tuple[str, ...] = (
        "a photo of a meteorite",
        "a photo of a meteorite rock",
        "a photo of a stony meteorite",
        "a photo of an iron meteorite",
        #"a photo of a meteorite specimen",
    )


def list_images(image_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(exts)]
    return sorted(files)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def build_text_features(model, tokenizer, class0_prompts, class1_prompts, device):
    prompts = list(class0_prompts) + list(class1_prompts)

    text_inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    text_outputs = model.get_text_features(**text_inputs)

    if isinstance(text_outputs, torch.Tensor):
        text_features = text_outputs
    elif hasattr(text_outputs, "text_embeds") and text_outputs.text_embeds is not None:
        text_features = text_outputs.text_embeds
    elif hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
        pooled = text_outputs.pooler_output

        if hasattr(model, "text_projection") and model.text_projection is not None:
            text_features = model.text_projection(pooled)
        else:
            text_features = pooled
    elif hasattr(text_outputs, "last_hidden_state") and text_outputs.last_hidden_state is not None:
        pooled = text_outputs.last_hidden_state[:, 0]

        if hasattr(model, "text_projection") and model.text_projection is not None:
            text_features = model.text_projection(pooled)
        else:
            text_features = pooled
    else:
        raise TypeError(f"Unsupported text output type: {type(text_outputs)}")

    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    n0 = len(class0_prompts)
    class0_feature = text_features[:n0].mean(dim=0)
    class1_feature = text_features[n0:].mean(dim=0)

    class_features = torch.stack([class0_feature, class1_feature], dim=0)
    class_features = class_features / class_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    return class_features


@torch.no_grad()
def encode_image_features(model, pixel_values):
    try:
        image_outputs = model.get_image_features(
            pixel_values=pixel_values,
            interpolate_pos_encoding=True,
        )
    except TypeError:
        image_outputs = model.get_image_features(
            pixel_values=pixel_values,
        )

    # 情况 1：已经直接返回最终 image feature
    if isinstance(image_outputs, torch.Tensor):
        image_features = image_outputs

    # 情况 2：有 image_embeds，通常也是最终 projection 后特征
    elif hasattr(image_outputs, "image_embeds") and image_outputs.image_embeds is not None:
        image_features = image_outputs.image_embeds

    # 情况 3：返回 pooler_output
    elif hasattr(image_outputs, "pooler_output") and image_outputs.pooler_output is not None:
        pooled = image_outputs.pooler_output

        proj_dim = model.config.projection_dim
        vision_hidden_dim = model.config.vision_config.hidden_size

        if pooled.shape[-1] == proj_dim:
            image_features = pooled
        elif pooled.shape[-1] == vision_hidden_dim:
            image_features = model.visual_projection(pooled)
        else:
            raise RuntimeError(
                f"Unexpected pooled dim: {pooled.shape[-1]}, "
                f"expected {proj_dim} or {vision_hidden_dim}."
            )

    # 情况 4：返回 last_hidden_state
    elif hasattr(image_outputs, "last_hidden_state") and image_outputs.last_hidden_state is not None:
        pooled = image_outputs.last_hidden_state[:, 0]

        proj_dim = model.config.projection_dim
        vision_hidden_dim = model.config.vision_config.hidden_size

        if pooled.shape[-1] == proj_dim:
            image_features = pooled
        elif pooled.shape[-1] == vision_hidden_dim:
            image_features = model.visual_projection(pooled)
        else:
            raise RuntimeError(
                f"Unexpected pooled dim: {pooled.shape[-1]}, "
                f"expected {proj_dim} or {vision_hidden_dim}."
            )

    else:
        raise TypeError(f"Unsupported image output type: {type(image_outputs)}")

    image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return image_features


@torch.no_grad()
def predict_clip_zeroshot(cfg: Config) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained(cfg.clip_model_dir, local_files_only=True).to(device)
    processor = CLIPImageProcessor.from_pretrained(cfg.clip_model_dir, local_files_only=True)
    tokenizer = CLIPTokenizerFast.from_pretrained(cfg.clip_model_dir, local_files_only=True)
    model.eval()

    text_features = build_text_features(
        model=model,
        tokenizer=tokenizer,
        class0_prompts=cfg.class0_prompts,
        class1_prompts=cfg.class1_prompts,
        device=device,
    )

    test_dir = os.path.join(cfg.dataset_root, "test_images")
    image_ids = list_images(test_dir)
    print(f"Test images: {len(image_ids)}")

    rows = []
    for start in tqdm(range(0, len(image_ids), cfg.batch_size), desc="Zero-shot predict"):
        batch_ids = image_ids[start:start + cfg.batch_size]
        images = [load_image(os.path.join(test_dir, img_id)) for img_id in batch_ids]

        inputs = processor(
            images=images,
            return_tensors="pt",
            size={"shortest_edge": cfg.image_size},
            crop_size={"height": cfg.image_size, "width": cfg.image_size},
        ).to(device)

        pixel_values = inputs["pixel_values"].to(device)
        image_features = encode_image_features(model, pixel_values)
        logits = image_features @ text_features.t()
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

        for img_id, prob in zip(batch_ids, probs):
            rows.append({"id": img_id, "prob": float(prob)})

    return pd.DataFrame(rows)


def apply_top_ratio(pred_df: pd.DataFrame, positive_ratio: float) -> pd.DataFrame:
    pred_df = pred_df.copy()
    n_total = len(pred_df)
    n_pos = int(round(n_total * positive_ratio))

    pred_df = pred_df.sort_values("prob", ascending=False).reset_index(drop=True)
    pred_df["rank"] = np.arange(1, n_total + 1)
    pred_df["label"] = 0
    pred_df.loc[:n_pos - 1, "label"] = 1

    print(f"Predicted positives: {n_pos}/{n_total} = {n_pos / n_total:.4f}")
    return pred_df


def make_submission(pred_df: pd.DataFrame, cfg: Config) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    prob_path = os.path.join(cfg.output_dir, "test_probs.csv")
    pred_df.to_csv(prob_path, index=False)
    print(f"Saved probabilities to: {prob_path}")

    sample_path = os.path.join(cfg.dataset_root, "sample_submission.csv")
    sample_df = pd.read_csv(sample_path)

    submission = sample_df[["id"]].copy()
    label_map = dict(zip(pred_df["id"], pred_df["label"]))
    submission["label"] = submission["id"].map(label_map)

    if submission["label"].isna().any():
        missing = submission.loc[submission["label"].isna(), "id"].head(5).tolist()
        raise RuntimeError(f"Missing predictions for ids, examples: {missing}")

    submission["label"] = submission["label"].astype(int)
    out_path = os.path.join(cfg.output_dir, "submission.csv")
    submission.to_csv(out_path, index=False)

    print(f"Saved submission to: {out_path}")
    print(submission["label"].value_counts().sort_index().to_dict())


def main():
    cfg = Config()
    pred_df = predict_clip_zeroshot(cfg)
    pred_df = apply_top_ratio(pred_df, cfg.target_positive_ratio)
    make_submission(pred_df, cfg)


if __name__ == "__main__":
    main()
