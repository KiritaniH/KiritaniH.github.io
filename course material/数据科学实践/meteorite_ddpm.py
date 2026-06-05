import os
import math
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def exists(x):
    return x is not None


# =========================
# Dataset
# =========================

def crop_meteorite_region(
    img: Image.Image,
    white_threshold: int = 245,
    margin_ratio: float = 0.06,
) -> Image.Image:
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        arr = np.stack([arr] * 3, axis=-1)

    # foreground mask: anything not close to pure white
    fg_mask = np.any(arr < white_threshold, axis=2)

    # fallback: if detection fails, return original image
    if not fg_mask.any():
        return img

    ys, xs = np.where(fg_mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    h, w = arr.shape[:2]
    box_h = y_max - y_min + 1
    box_w = x_max - x_min + 1

    margin_y = max(2, int(box_h * margin_ratio))
    margin_x = max(2, int(box_w * margin_ratio))

    y_min = max(0, y_min - margin_y)
    y_max = min(h - 1, y_max + margin_y)
    x_min = max(0, x_min - margin_x)
    x_max = min(w - 1, x_max + margin_x)

    return img.crop((x_min, y_min, x_max + 1, y_max + 1))


def resize_and_pad_to_square(
    img: Image.Image,
    image_size: int,
    fill_color: Tuple[int, int, int] = (255, 255, 255),
    object_ratio_range: Tuple[float, float] = (0.60, 0.78),
    random_translate: bool = True,
) -> Image.Image:
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGB", (image_size, image_size), fill_color)

    min_ratio, max_ratio = object_ratio_range
    ratio = random.uniform(min_ratio, max_ratio)

    target_max_side = max(1, int(image_size * ratio))
    scale = target_max_side / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img = img.resize((new_w, new_h), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (image_size, image_size), fill_color)

    max_left = max(0, image_size - new_w)
    max_top = max(0, image_size - new_h)

    if random_translate:
        left = random.randint(0, max_left) if max_left > 0 else 0
        top = random.randint(0, max_top) if max_top > 0 else 0
    else:
        left = max_left // 2
        top = max_top // 2

    canvas.paste(img, (left, top))
    return canvas


class MeteoriteDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 64,
        crop_foreground: bool = True,
        white_threshold: int = 245,
        margin_ratio: float = 0.08,
        object_ratio_range: Tuple[float, float] = (0.60, 0.78),
        random_translate: bool = True,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]:
            self.paths.extend(self.root.glob(ext))
        self.paths = sorted(self.paths)

        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found in {root}")

        self.image_size = image_size
        self.crop_foreground = crop_foreground
        self.white_threshold = white_threshold
        self.margin_ratio = margin_ratio
        self.object_ratio_range = object_ratio_range
        self.random_translate = random_translate
        self.augment = augment

        self.augment_transform = None #transforms.Compose([
            #transforms.RandomHorizontalFlip(p=0.5),
            
            #transforms.RandomRotation(
                #degrees=10,
                #interpolation=transforms.InterpolationMode.BILINEAR,
                #fill=(255, 255, 255),
            #),
            
        #])


        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),                      # [0, 1]
            transforms.Lambda(lambda x: x * 2.0 - 1.0) # [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def preprocess_image(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")

        if self.crop_foreground:
            img = crop_meteorite_region(
                img,
                white_threshold=self.white_threshold,
                margin_ratio=self.margin_ratio,
            )

        img = resize_and_pad_to_square(
            img,
            image_size=self.image_size,
            fill_color=(255, 255, 255),
            object_ratio_range=self.object_ratio_range,
            random_translate=self.random_translate,
        )

        if self.augment:
            img = self.augment_transform(img)
            
        return img

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.preprocess_image(img)
        img = self.tensor_transform(img)
        return img

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = time[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb
        
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)
        x = x.view(b, c, h * w).permute(0, 2, 1)   # [B, HW, C]
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x.permute(0, 2, 1).view(b, c, h, w)

        return x + residual


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, time_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.conv(x, t)
        return self.down(h), h


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, t)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = DownBlock(base_channels, base_channels, time_dim)
        self.down2 = DownBlock(base_channels, base_channels * 2, time_dim)
        self.attn16 = AttentionBlock(base_channels * 2)

        self.down3 = DownBlock(base_channels * 2, base_channels * 4, time_dim)

        self.mid1 = ConvBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid_attn = AttentionBlock(base_channels * 4)
        self.mid2 = ConvBlock(base_channels * 4, base_channels * 4, time_dim)

        self.up1 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, time_dim)
        self.up_attn16 = AttentionBlock(base_channels * 2)

        self.up2 = UpBlock(base_channels * 2, base_channels * 2, base_channels, time_dim)
        self.up3 = UpBlock(base_channels, base_channels, base_channels, time_dim)

        self.out = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 1)
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        t = self.time_embed(time)

        x = self.init_conv(x)

        x, skip1 = self.down1(x, t)
        x, skip2 = self.down2(x, t)
        x = self.attn16(x)

        x, skip3 = self.down3(x, t)

        x = self.mid1(x, t)
        x = self.mid_attn(x)
        x = self.mid2(x, t)

        x = self.up1(x, skip3, t)
        x = self.up_attn16(x)

        x = self.up2(x, skip2, t)
        x = self.up3(x, skip1, t)

        return self.out(x)

def cosine_beta_schedule(timesteps: int, s: float = 0.008, device: str = "cuda"):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)

    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, min=1e-5, max=0.999)
    return betas


class DDPM:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cuda",
        schedule: str = "cosine",
    ):
        self.timesteps = timesteps
        self.device = device

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps, device=device)
        else:
            betas = torch.linspace(beta_start, beta_end, timesteps, device=device)

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat(
            [torch.ones(1, device=device), alpha_bars[:-1]],
            dim=0
        )

        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.alpha_bars_prev = alpha_bars_prev
        self.posterior_variance = posterior_variance

        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        
    def extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape):
        out = arr.gather(0, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_1mab = self.extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)
        xt = sqrt_ab * x0 + sqrt_1mab * noise
        return xt, noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor):
        beta_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alpha_bar_t = self.extract(self.sqrt_one_minus_alpha_bars, t, x.shape)
        sqrt_recip_alpha_t = self.extract(self.sqrt_recip_alphas, t, x.shape)
        posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)

        pred_noise = model(x, t)
        model_mean = sqrt_recip_alpha_t * (
            x - (beta_t / sqrt_one_minus_alpha_bar_t) * pred_noise
        )

        nonzero_mask = (t != 0).float().view(x.shape[0], *((1,) * (len(x.shape) - 1)))
        noise = torch.randn_like(x)
        sigma = torch.sqrt(torch.clamp(posterior_variance_t, min=1e-20))

        return model_mean + nonzero_mask * sigma * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, shape):
        model.eval()
        x = torch.randn(shape, device=self.device)
        for step in tqdm(reversed(range(self.timesteps)), total=self.timesteps, desc="Sampling"):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        x = torch.clamp(x, -1.0, 1.0)
        return x


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


def save_checkpoint(model, ema, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "ema": ema.shadow,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }, path)


def load_checkpoint(model, optimizer, path, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def train(
    data_dir: str = "./meteorite",
    image_size: int = 64,
    batch_size: int = 32,
    num_epochs: int = 100,
    lr: float = 2e-4,
    timesteps: int = 1000,
    #base_channels: int = 64,
    ema_decay: float = 0.9999,
    num_workers: int = 4,
    seed: int = 42,
    checkpoint_dir: str = "./checkpoints",
    sample_dir: str = "./train_samples",
    resume_path: Optional[str] = None,
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = MeteoriteDataset(
        data_dir,
        image_size=image_size,
        crop_foreground=True,
        white_threshold=245,
        margin_ratio=0.08,
        object_ratio_range=(0.60, 0.78),
        random_translate=True,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )
    criterion = nn.MSELoss()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=timesteps,
        beta_schedule="squaredcos_cap_v2",
        #prediction_type="epsilon",
        prediction_type="v_prediction",
    )

    ema = EMA(model, decay=ema_decay)

    start_epoch = 0
    if exists(resume_path):
        ckpt = load_checkpoint(model, optimizer, resume_path, device=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        if "ema" in ckpt:
            ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for x0 in pbar:
            x0 = x0.to(device)

            t = torch.randint(0, timesteps, (x0.size(0),), device=device).long()
            noise = torch.randn_like(x0)
            xt = noise_scheduler.add_noise(x0, noise, t)

            #pred_noise = model(xt, t).sample
            #loss = criterion(pred_noise, noise)

            target = noise_scheduler.get_velocity(x0, noise, t)
            pred = model(xt, t).sample
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.6f}")

        lr_scheduler.step()

        if (epoch + 1) % 20 == 0 or (epoch + 1) == num_epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"ddpm_epoch_{epoch + 1}.pt")
            save_checkpoint(model, ema, optimizer, epoch, ckpt_path)

        # save periodic visualization samples with EMA weights
        ema_model = UNet2DModel(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        ).to(device)

        ema_model.load_state_dict(model.state_dict())
        ema.copy_to(ema_model)

        sample_scheduler = DDIMScheduler(
            num_train_timesteps=timesteps,
            beta_schedule="squaredcos_cap_v2",
            #prediction_type="epsilon",
            prediction_type="v_prediction",
        )
        sample_scheduler.set_timesteps(250)

        x = torch.randn((16, 3, image_size, image_size), device=device)
        ema_model.eval()
        for step in sample_scheduler.timesteps:
            with torch.no_grad():
                pred = ema_model(x, step).sample
            x = sample_scheduler.step(pred, step, x).prev_sample

        samples = torch.clamp(x, -1.0, 1.0)
        samples = (samples + 1.0) / 2.0
        save_image(samples, os.path.join(sample_dir, f"epoch_{epoch + 1:03d}.png"), nrow=4)
    final_ckpt = os.path.join(checkpoint_dir, "ddpm_final.pt")
    save_checkpoint(model, ema, optimizer, num_epochs - 1, final_ckpt)
    print(f"Training finished. Final checkpoint saved to {final_ckpt}")


@torch.no_grad()
def generate_images(
    checkpoint_path: str,
    output_dir: str = "./generated_pictures",
    num_images: int = 1000,
    image_size: int = 64,
    timesteps: int = 1000,
    batch_size: int = 32,
    ddim_steps: int = 250,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    if "ema" in ckpt:
        for name, param in model.named_parameters():
            if param.requires_grad and name in ckpt["ema"]:
                param.data.copy_(ckpt["ema"][name].to(device))

    sampler = DDIMScheduler(
        num_train_timesteps=timesteps,
        beta_schedule="squaredcos_cap_v2",
        #prediction_type="epsilon",
        prediction_type="v_prediction",
    )
    sampler.set_timesteps(ddim_steps)

    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    model.eval()
    while saved < num_images:
        cur_bs = min(batch_size, num_images - saved)
        x = torch.randn((cur_bs, 3, image_size, image_size), device=device)

        for step in sampler.timesteps:
            pred = model(x, step).sample
            x = sampler.step(pred, step, x).prev_sample

        x = torch.clamp(x, -1.0, 1.0)
        x = (x + 1.0) / 2.0

        for i in range(cur_bs):
            save_path = os.path.join(output_dir, f"{saved:04d}.png")
            save_image(x[i], save_path)
            saved += 1

    print(f"Saved {num_images} images to {output_dir}")

# test
@torch.no_grad()
def preview_preprocessing(
    data_dir: str = "./meteorite",
    output_path: str = "./preprocessed_preview.png",
    image_size: int = 64,
    num_images: int = 16,
):
    dataset = MeteoriteDataset(
        data_dir,
        image_size=image_size,
        crop_foreground=True,
        white_threshold=245,
        margin_ratio=0.08,
        object_ratio_range=(0.60, 0.78),
        random_translate=True,
    )
    loader = DataLoader(dataset, batch_size=num_images, shuffle=False)
    x = next(iter(loader))   # [-1, 1]
    x = (x + 1.0) / 2.0      # [0, 1]
    save_image(x, output_path, nrow=4)
    print(f"Saved preprocessing preview to {output_path}")


if __name__ == "__main__":
    '''
    preview_preprocessing(
        data_dir="./meteorite",
        output_path="./preprocessed_preview.png",
        image_size=64,
        num_images=16,
    )
    '''
    train(
        data_dir="./meteorite",
        image_size=256,
        batch_size=8,
        num_epochs=360,
        lr=5e-6,
        timesteps=1000,
        resume_path="./checkpoints/ddpm_final.pt",
        #base_channels=128,
    )

    generate_images(
        checkpoint_path="./checkpoints/ddpm_final.pt",
        output_dir="./generated_pictures",
        num_images=1000,
        image_size=256,
        batch_size=8,
        timesteps=1000,
        ddim_steps=200,
        #base_channels=128,
    )
    pass
