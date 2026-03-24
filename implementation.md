# Soft-constrained Schrödinger Bridges — Exact Implementation Steps

> Video frame interpolation via stochastic optimal transport with soft CLIP-guided constraints.

---

## Prerequisites

```bash
# Python environment
python -m venv schrodinger_env
source schrodinger_env/bin/activate

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install clip-by-openai einops wandb torchmetrics
pip install onnx onnxruntime

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Phase 1: Data Engineering

### Step 1.1 — Download Vimeo-90K triplets

```bash
# ~32GB download
wget http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip
unzip vimeo_triplet.zip -d ./data/
# Structure: data/vimeo_triplet/sequences/<folder>/<clip>/im1.png im2.png im3.png
```

### Step 1.2 — Dataset class

```python
# dataset.py
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import clip

class VimeoTripletDataset(Dataset):
    def __init__(self, root, split="train", size=128, prompts=None):
        self.root = Path(root)
        self.size = size
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),   # maps [0,1] → [-1,1]
        ])
        split_file = self.root / f"tri_{split}list.txt"
        with open(split_file) as f:
            self.clips = [l.strip() for l in f.readlines()]

        # Default prompts pool — will be sampled per batch
        self.prompts = prompts or [
            "a warm sunset scene",
            "a scene with cool blue tones",
            "a bright vivid image",
            "a dark moody scene",
        ]

        self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
        self.clip_model.eval()
        self._encode_prompts()

    def _encode_prompts(self):
        tokens = clip.tokenize(self.prompts)
        with torch.no_grad():
            self.clip_embeddings = self.clip_model.encode_text(tokens)  # (P, 512)
            self.clip_embeddings = self.clip_embeddings / self.clip_embeddings.norm(dim=-1, keepdim=True)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_path = self.root / "sequences" / self.clips[idx]
        x0 = self.transform(Image.open(clip_path / "im1.png").convert("RGB"))
        x1 = self.transform(Image.open(clip_path / "im3.png").convert("RGB"))
        gt = self.transform(Image.open(clip_path / "im2.png").convert("RGB"))
        # Pick a random prompt embedding as soft constraint
        c_idx = torch.randint(len(self.prompts), (1,)).item()
        c = self.clip_embeddings[c_idx]  # (512,)
        return x0, x1, c, gt
```

### Step 1.3 — DataLoader setup

```python
from torch.utils.data import DataLoader

train_ds = VimeoTripletDataset("./data/vimeo_triplet", split="train",  size=128)
val_ds   = VimeoTripletDataset("./data/vimeo_triplet", split="test",   size=128)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=8,  shuffle=False, num_workers=4, pin_memory=True)
```

---

## Phase 2: The Bridge Engine

### Step 2.1 — Brownian Bridge sampler

```python
# bridge.py
import torch
import torch.nn.functional as F
import math

def brownian_bridge(x0: torch.Tensor, x1: torch.Tensor,
                    t: torch.Tensor, sigma: float = 0.1):
    """
    Sample a noisy intermediate frame along the Brownian Bridge.

    x0, x1 : (B, C, H, W)  — start and end frames, normalised to [-1, 1]
    t       : (B,)          — timestep in [0, 1]
    sigma   : float         — noise scale

    Returns
    -------
    xt  : (B, C, H, W)   — noisy sample at time t
    eps : (B, C, H, W)   — the noise drawn (needed for velocity target)
    """
    t = t.view(-1, 1, 1, 1).clamp(0.01, 0.99)
    eps = torch.randn_like(x0)
    noise_scale = sigma * torch.sqrt(t * (1 - t))  # peaks at t=0.5, zero at endpoints
    xt = (1 - t) * x0 + t * x1 + noise_scale * eps
    return xt, eps


def velocity_target(x0: torch.Tensor, x1: torch.Tensor,
                    t: torch.Tensor, eps: torch.Tensor,
                    sigma: float = 0.1):
    """
    Compute the exact instantaneous velocity of the bridge path.
    This is what the neural network must learn to predict.

    Formula: u_t = (x1 - x0) + sigma * (1 - 2t) / (2 * sqrt(t(1-t))) * eps
    """
    t = t.view(-1, 1, 1, 1).clamp(0.01, 0.99)
    drift = x1 - x0
    correction_scale = sigma * (1 - 2 * t) / (2 * torch.sqrt(t * (1 - t)))
    return drift + correction_scale * eps
```

### Step 2.2 — Sanity checks (run before training)

```python
# test_bridge.py
import torch
from bridge import brownian_bridge, velocity_target

def test_bridge_mean():
    """Mean of x_t should be the linear interpolant (1-t)x0 + t*x1."""
    x0 = torch.zeros(1, 3, 8, 8)
    x1 = torch.ones(1, 3, 8, 8)
    t  = torch.tensor([0.5])

    samples = torch.stack([brownian_bridge(x0, x1, t)[0] for _ in range(2000)])
    expected_mean = 0.5 * (x0 + x1)

    assert samples.mean(0).allclose(expected_mean, atol=0.05), "Mean check failed"
    print("PASS: bridge mean")

def test_bridge_variance():
    """Var(x_t) should be sigma^2 * t * (1 - t)."""
    sigma = 0.1
    x0 = torch.zeros(1, 3, 8, 8)
    x1 = torch.zeros(1, 3, 8, 8)
    t  = torch.tensor([0.5])

    samples = torch.stack([brownian_bridge(x0, x1, t, sigma=sigma)[0] for _ in range(2000)])
    expected_var = sigma**2 * 0.5 * 0.5

    assert abs(samples.var().item() - expected_var) < 0.002, "Variance check failed"
    print("PASS: bridge variance")

def test_endpoints():
    """At t=0.01 and t=0.99 the noise should be near-zero."""
    x0 = torch.zeros(1, 3, 8, 8)
    x1 = torch.ones(1, 3, 8, 8)

    xt_start, _ = brownian_bridge(x0, x1, torch.tensor([0.01]))
    xt_end, _   = brownian_bridge(x0, x1, torch.tensor([0.99]))

    assert xt_start.allclose(x0, atol=0.05), "Start endpoint check failed"
    assert xt_end.allclose(x1, atol=0.05),   "End endpoint check failed"
    print("PASS: endpoints")

if __name__ == "__main__":
    test_bridge_mean()
    test_bridge_variance()
    test_endpoints()
    print("All bridge tests passed.")
```

---

## Phase 3: The UNet Architecture

### Step 3.1 — Building blocks

```python
# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def time_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    """
    Sinusoidal time embedding, identical to DDPM.
    t   : (B,)
    out : (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """ResNet block with AdaGN time conditioning."""
    def __init__(self, in_ch, out_ch, time_dim=256):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)   # scale + shift
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        scale, shift = self.time_proj(F.silu(t_emb)).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.conv1(F.silu(self.norm1(x)))
        h = h * (1 + scale) + shift          # AdaGN: modulate by time
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for spatial feature maps."""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)   # (B, HW, C)
        h, _ = self.attn(h, h, h)
        return x + h.transpose(1, 2).view(B, C, H, W)


class CrossAttention(nn.Module):
    """Cross-attention: spatial features query a CLIP context vector."""
    def __init__(self, channels, context_dim=512, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(context_dim, channels)
        self.to_v = nn.Linear(context_dim, channels)
        self.out  = nn.Linear(channels, channels)
        self.num_heads = num_heads

    def forward(self, x, c):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)   # (B, HW, C)
        context = c.unsqueeze(1)                               # (B, 1, 512)
        q = self.to_q(h)
        k = self.to_k(context)
        v = self.to_v(context)
        out = F.scaled_dot_product_attention(q, k, v)
        out = self.out(out)
        return x + out.transpose(1, 2).view(B, C, H, W)
```

### Step 3.2 — Full UNet

```python
class UNet(nn.Module):
    """
    UNet velocity predictor for the Schrödinger Bridge.

    Input  : cat(x_t, x0) → 6 channels
    Output : predicted velocity field, same shape as x_t
    """
    def __init__(self, in_ch=6, model_ch=128, ch_mult=(1,2,4,8),
                 time_dim=256, context_dim=512):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.time_dim = time_dim

        channels = [model_ch * m for m in ch_mult]   # [128, 256, 512, 1024]

        # Encoder
        self.input_conv = nn.Conv2d(in_ch, channels[0], 3, padding=1)
        self.down0 = nn.ModuleList([ResBlock(channels[0], channels[0], time_dim),
                                    ResBlock(channels[0], channels[0], time_dim)])
        self.down1 = nn.ModuleList([ResBlock(channels[0], channels[1], time_dim),
                                    ResBlock(channels[1], channels[1], time_dim)])
        self.down2 = nn.ModuleList([ResBlock(channels[1], channels[2], time_dim),
                                    ResBlock(channels[2], channels[2], time_dim),
                                    SelfAttention(channels[2])])
        self.down3 = nn.ModuleList([ResBlock(channels[2], channels[3], time_dim),
                                    ResBlock(channels[3], channels[3], time_dim),
                                    SelfAttention(channels[3])])

        self.pool = nn.AvgPool2d(2)

        # Bottleneck
        self.mid_res1 = ResBlock(channels[3], channels[3], time_dim)
        self.mid_self_attn = SelfAttention(channels[3])
        self.mid_cross_attn = CrossAttention(channels[3], context_dim)
        self.mid_res2 = ResBlock(channels[3], channels[3], time_dim)

        # Decoder (channels doubled by skip connections)
        self.up3 = nn.ModuleList([ResBlock(channels[3]*2, channels[2], time_dim),
                                   ResBlock(channels[2],   channels[2], time_dim),
                                   SelfAttention(channels[2])])
        self.up2 = nn.ModuleList([ResBlock(channels[2]*2, channels[1], time_dim),
                                   ResBlock(channels[1],   channels[1], time_dim),
                                   SelfAttention(channels[1])])
        self.up1 = nn.ModuleList([ResBlock(channels[1]*2, channels[0], time_dim),
                                   ResBlock(channels[0],   channels[0], time_dim)])
        self.up0 = nn.ModuleList([ResBlock(channels[0]*2, channels[0], time_dim),
                                   ResBlock(channels[0],   channels[0], time_dim)])

        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")

        self.output_norm = nn.GroupNorm(8, channels[0])
        self.output_conv = nn.Conv2d(channels[0], 3, 3, padding=1)

    def forward(self, xt, t, x0, c):
        """
        xt : (B, 3, H, W)    noisy frame at time t
        t  : (B,)            timestep in [0, 1]
        x0 : (B, 3, H, W)    conditioning frame
        c  : (B, 512)        CLIP soft constraint embedding
        """
        t_emb = self.time_embed(time_embedding(t, self.time_dim))
        x = torch.cat([xt, x0], dim=1)   # (B, 6, H, W)
        x = self.input_conv(x)

        # Encoder — save skip connections
        for blk in self.down0: x = blk(x, t_emb) if isinstance(blk, ResBlock) else blk(x)
        s0 = x; x = self.pool(x)
        for blk in self.down1: x = blk(x, t_emb) if isinstance(blk, ResBlock) else blk(x)
        s1 = x; x = self.pool(x)
        for blk in self.down2: x = blk(x, t_emb) if isinstance(blk, ResBlock) else blk(x)
        s2 = x; x = self.pool(x)
        for blk in self.down3: x = blk(x, t_emb) if isinstance(blk, ResBlock) else blk(x)
        s3 = x; x = self.pool(x)

        # Bottleneck
        x = self.mid_res1(x, t_emb)
        x = self.mid_self_attn(x)
        x = self.mid_cross_attn(x, c)   # inject CLIP constraint here
        x = self.mid_res2(x, t_emb)

        # Decoder
        x = self.up_sample(x); x = torch.cat([x, s3], dim=1)
        for blk in self.up3: x = blk(x, t_emb) if isinstance(blk, ResBlock) else blk(x)
        x = self.up_sample(x); x = torch.cat([x, s2], dim=1)
        for blk in self.up2: x = blk(x, t_emb) if isinstance(blk, ResBlock) else blk(x)
        x = self.up_sample(x); x = torch.cat([x, s1], dim=1)
        for blk in self.up1: x = blk(x, t_emb) if isinstance(blk, ResBlock) else blk(x)
        x = self.up_sample(x); x = torch.cat([x, s0], dim=1)
        for blk in self.up0: x = blk(x, t_emb) if isinstance(blk, ResBlock) else blk(x)

        return self.output_conv(F.silu(self.output_norm(x)))
```

### Step 3.3 — Verify model dimensions

```python
# Quick shape check — run before any training
model = UNet()
xt  = torch.randn(2, 3, 128, 128)
t   = torch.rand(2)
x0  = torch.randn(2, 3, 128, 128)
c   = torch.randn(2, 512)

with torch.no_grad():
    out = model(xt, t, x0, c)

assert out.shape == xt.shape, f"Shape mismatch: {out.shape} vs {xt.shape}"
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
# Expected: ~85M params
```

---

## Phase 4: Training Loop

### Step 4.1 — Guidance loss (the soft constraint)

```python
# losses.py
import torch
import torch.nn.functional as F
import clip


class GuidanceLoss:
    """
    Pulls the predicted velocity toward the CLIP edit direction.
    The CLIP model is frozen — do not fine-tune it.
    """
    def __init__(self, clip_model, device):
        self.clip_model = clip_model.to(device).eval()
        for p in self.clip_model.parameters():
            p.requires_grad_(False)

    def __call__(self, v_pred: torch.Tensor, xt: torch.Tensor,
                 c: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        # Step the frame forward by dt using predicted velocity
        x_next = (xt + v_pred * dt).clamp(-1, 1)

        # Resize to CLIP input size (224x224) and rescale to [0, 1]
        x_clip = F.interpolate(x_next.detach(), size=224, mode="bilinear",
                               align_corners=False)
        x_clip = (x_clip + 1) / 2   # [-1, 1] → [0, 1]

        # Normalise to CLIP's expected stats
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=x_clip.device).view(1,3,1,1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=x_clip.device).view(1,3,1,1)
        x_clip = (x_clip - mean) / std

        feat = self.clip_model.encode_image(x_clip)
        feat = feat / feat.norm(dim=-1, keepdim=True)

        # Cosine distance: 1 means orthogonal (worst), 0 means aligned (best)
        return (1 - F.cosine_similarity(feat, c)).mean()
```

### Step 4.2 — Lambda scheduler

```python
def get_lambda(step: int, total_steps: int,
               warmup_frac: float = 0.2,
               ramp_frac: float = 0.5,
               lambda_max: float = 0.1) -> float:
    """
    Phase 1 (0 → 20%): lambda = 0  — bridge loss only, establish valid path
    Phase 2 (20% → 50%): linear ramp 0 → lambda_max
    Phase 3 (50% → 100%): lambda = lambda_max
    """
    warmup = int(warmup_frac * total_steps)
    ramp_end = int(ramp_frac * total_steps)
    if step < warmup:
        return 0.0
    if step >= ramp_end:
        return lambda_max
    return lambda_max * (step - warmup) / (ramp_end - warmup)
```

### Step 4.3 — Full training script

```python
# train.py
import torch
import torch.nn.functional as F
import wandb
import clip
from torch.cuda.amp import GradScaler, autocast

from dataset import VimeoTripletDataset
from torch.utils.data import DataLoader
from bridge import brownian_bridge, velocity_target
from model import UNet
from losses import GuidanceLoss

# ── Config ──────────────────────────────────────────────────────────────
DEVICE      = "cuda"
BATCH_SIZE  = 16
LR          = 2e-4
WEIGHT_DECAY = 0.01
TOTAL_STEPS = 200_000
SIGMA       = 0.1
GRAD_CLIP   = 1.0
LOG_EVERY   = 100
SAVE_EVERY  = 5_000
# ────────────────────────────────────────────────────────────────────────

def train():
    wandb.init(project="schrodinger-bridge-video", config={
        "batch_size": BATCH_SIZE, "lr": LR, "sigma": SIGMA
    })

    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    guidance_loss_fn = GuidanceLoss(clip_model, DEVICE)

    train_ds = VimeoTripletDataset("./data/vimeo_triplet", split="train", size=128)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)

    model = UNet().to(DEVICE)
    model = torch.compile(model, mode="reduce-overhead")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)
    scaler = GradScaler()   # mixed precision

    step = 0
    while step < TOTAL_STEPS:
        for x0, x1, c, _ in train_loader:
            x0, x1, c = x0.to(DEVICE), x1.to(DEVICE), c.to(DEVICE)
            B = x0.shape[0]

            # Sample t — avoid endpoints due to velocity singularity
            t = torch.rand(B, device=DEVICE) * 0.98 + 0.01

            with autocast():
                xt, eps = brownian_bridge(x0, x1, t, sigma=SIGMA)
                u_true  = velocity_target(x0, x1, t, eps, sigma=SIGMA)

                v_pred = model(xt, t, x0, c)

                loss_bridge = F.mse_loss(v_pred, u_true)

                lam = get_lambda(step, TOTAL_STEPS)
                if lam > 0:
                    loss_soft = lam * guidance_loss_fn(v_pred, xt, c)
                else:
                    loss_soft = torch.tensor(0.0, device=DEVICE)

                loss = loss_bridge + loss_soft

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % LOG_EVERY == 0:
                wandb.log({
                    "loss/bridge":   loss_bridge.item(),
                    "loss/guidance": loss_soft.item() / max(lam, 1e-8),
                    "loss/total":    loss.item(),
                    "lambda":        lam,
                    "lr":            scheduler.get_last_lr()[0],
                    "step":          step,
                })
                print(f"Step {step:>7d}  bridge={loss_bridge:.4f}  "
                      f"guide={loss_soft:.4f}  λ={lam:.4f}")

            if step % SAVE_EVERY == 0 and step > 0:
                torch.save(model.state_dict(), f"checkpoints/step_{step:07d}.pt")

            step += 1
            if step >= TOTAL_STEPS:
                break

    torch.save(model.state_dict(), "checkpoints/final.pt")
    print("Training complete.")

if __name__ == "__main__":
    import os
    os.makedirs("checkpoints", exist_ok=True)
    train()
```

### Step 4.4 — What to monitor in wandb

| Metric                            | Healthy range                 | Warning sign                               |
| --------------------------------- | ----------------------------- | ------------------------------------------ |
| `loss/bridge`                     | Decreasing from ~0.5 to <0.05 | Flat after 10k steps → check LR            |
| `loss/guidance` (normalised by λ) | Slowly decreasing             | Spikes dominating bridge loss → λ too high |
| `lambda`                          | 0 → 0.1 linear ramp           | —                                          |
| Grad norm                         | <1.0 after clipping           | Consistently hitting clip → lower LR       |

---

## Phase 5: Inference (SDE Solver)

### Step 5.1 — Euler-Maruyama integrator

```python
# inference.py
import torch
import torch.nn.functional as F
import math

@torch.no_grad()
def generate(x0: torch.Tensor, x1: torch.Tensor, c: torch.Tensor,
             model, N: int = 20, sigma: float = 0.1,
             device: str = "cuda") -> torch.Tensor:
    """
    Walk from x0 at t=0 to a sample near x1 at t=1.

    x0, x1  : (B, 3, H, W)
    c        : (B, 512)  CLIP soft constraint
    N        : number of Euler steps (5=fast, 20=quality)

    Returns the generated middle frame : (B, 3, H, W)
    """
    x = x0.clone().to(device)
    dt = 1.0 / N

    for i in range(N):
        # Evaluate velocity at midpoint of interval (midpoint rule)
        t_val = (i + 0.5) * dt
        t = torch.full((x.shape[0],), t_val, device=device)

        v = model(x, t, x0, c)

        # Stochastic kick: σ(t) * dW_t
        dW = torch.randn_like(x) * sigma * math.sqrt(dt)

        x = x + v * dt + dW

    return x.clamp(-1, 1)
```

### Step 5.2 — Evaluation metrics

```python
# eval.py
import torch
import torch.nn.functional as F
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

def evaluate(model, val_loader, device, N=20, sigma=0.1, max_batches=50):
    model.eval()
    psnrs, ssims, swds = [], [], []

    for i, (x0, x1, c, gt) in enumerate(val_loader):
        if i >= max_batches:
            break
        x0, x1, c, gt = x0.to(device), x1.to(device), c.to(device), gt.to(device)

        pred = generate(x0, x1, c, model, N=N, sigma=sigma, device=device)

        # Rescale [-1,1] → [0,1] for metrics
        pred_01 = (pred + 1) / 2
        gt_01   = (gt   + 1) / 2

        psnrs.append(peak_signal_noise_ratio(pred_01, gt_01, data_range=1.0).item())
        ssims.append(structural_similarity_index_measure(pred_01, gt_01, data_range=1.0).item())
        swds.append(sliced_wasserstein(pred_01, gt_01).item())

    print(f"PSNR:  {sum(psnrs)/len(psnrs):.2f} dB")
    print(f"SSIM:  {sum(ssims)/len(ssims):.4f}")
    print(f"SWD:   {sum(swds)/len(swds):.4f}")


def sliced_wasserstein(real: torch.Tensor, fake: torch.Tensor,
                       n_proj: int = 512) -> torch.Tensor:
    """
    Sliced Wasserstein Distance — scalable proxy for W2 between image distributions.
    Flatten spatial dims and project onto random 1D directions.
    """
    B = real.shape[0]
    r = real.view(B, -1)   # (B, C*H*W)
    f = fake.view(B, -1)

    projs = F.normalize(torch.randn(n_proj, r.shape[-1], device=r.device), dim=-1)
    r_proj = (r @ projs.T).sort(dim=0).values   # (B, n_proj)
    f_proj = (f @ projs.T).sort(dim=0).values

    return (r_proj - f_proj).pow(2).mean().sqrt()
```

### Step 5.3 — Quick visual check

```python
import torchvision.utils as vutils

# Load one batch and generate
x0, x1, c, gt = next(iter(val_loader))
pred = generate(x0[:4], x1[:4], c[:4], model)

# Save side-by-side: x0 | pred | gt | x1
grid = vutils.make_grid(
    torch.cat([(x0[:4]+1)/2, (pred+1)/2, (gt[:4]+1)/2, (x1[:4]+1)/2], dim=0),
    nrow=4, padding=4
)
vutils.save_image(grid, "sample_output.png")
```

---

## Phase 6: WASM Deployment

### Step 6.1 — Export to ONNX

```python
# export_onnx.py
import torch
from model import UNet

# Load checkpoint into a fresh (non-compiled) model instance
model = UNet()
model.load_state_dict(torch.load("checkpoints/final.pt", map_location="cpu"))
model.eval()

# Dummy inputs matching training shapes
xt_d = torch.randn(1, 3, 128, 128)
t_d  = torch.rand(1)
x0_d = torch.randn(1, 3, 128, 128)
c_d  = torch.randn(1, 512)

torch.onnx.export(
    model,
    args=(xt_d, t_d, x0_d, c_d),
    f="bridge_unet.onnx",
    input_names=["xt", "t", "x0", "c"],
    output_names=["velocity"],
    dynamic_axes={
        "xt":       {0: "batch"},
        "t":        {0: "batch"},
        "x0":       {0: "batch"},
        "c":        {0: "batch"},
        "velocity": {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
)
print("Exported bridge_unet.onnx")

# Verify the export
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("bridge_unet.onnx")
out = sess.run(None, {
    "xt": xt_d.numpy(), "t": t_d.numpy(),
    "x0": x0_d.numpy(), "c": c_d.numpy()
})
print(f"ONNX output shape: {out[0].shape}")   # Should be (1, 3, 128, 128)
```

### Step 6.2 — Euler-Maruyama solver in Rust

```toml
# Cargo.toml
[package]
name = "schrodinger-solver"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2"
js-sys = "0.3"
rand = { version = "0.8", features = ["small_rng"] }

[profile.release]
opt-level = 3
lto = true
```

```rust
// src/lib.rs
use wasm_bindgen::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;

#[wasm_bindgen]
pub struct BridgeSolver {
    n_steps: usize,
    sigma:   f32,
}

#[wasm_bindgen]
impl BridgeSolver {
    #[wasm_bindgen(constructor)]
    pub fn new(n_steps: usize, sigma: f32) -> Self {
        BridgeSolver { n_steps, sigma }
    }

    /// One Euler-Maruyama step.
    /// velocity_fn is a JS callback: (flat_xt, t_scalar) -> flat_velocity
    pub fn step(&self, xt: &[f32], velocity: &[f32], t: f32, dt: f32) -> Vec<f32> {
        let mut rng = SmallRng::from_entropy();
        let noise_scale = self.sigma * dt.sqrt();
        xt.iter()
            .zip(velocity.iter())
            .map(|(xi, vi)| {
                let dw: f32 = rng.sample(StandardNormal);
                (xi + vi * dt + dw * noise_scale).clamp(-1.0, 1.0)
            })
            .collect()
    }

    pub fn dt(&self) -> f32 {
        1.0 / self.n_steps as f32
    }
}
```

```bash
# Compile to WASM
wasm-pack build --target web --release
# Output: pkg/schrodinger_solver_bg.wasm  +  pkg/schrodinger_solver.js
```

### Step 6.3 — Integrate in the browser

```javascript
// bridge_inference.js
import init, { BridgeSolver } from "./pkg/schrodinger_solver.js";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.esm.min.js";

async function runBridge(x0Flat, x1Flat, cFlat, H = 128, W = 128, N = 10) {
	await init(); // init WASM module

	const session = await ort.InferenceSession.create("./bridge_unet.onnx", {
		executionProviders: ["wasm"],
	});

	const solver = new BridgeSolver(N, 0.1);
	const dt = solver.dt();

	let xt = Float32Array.from(x0Flat);

	for (let i = 0; i < N; i++) {
		const t_val = (i + 0.5) * dt;

		const feeds = {
			xt: new ort.Tensor("float32", xt, [1, 3, H, W]),
			t: new ort.Tensor("float32", [t_val], [1]),
			x0: new ort.Tensor("float32", x0Flat, [1, 3, H, W]),
			c: new ort.Tensor("float32", cFlat, [1, 512]),
		};

		const results = await session.run(feeds);
		const velocity = results["velocity"].data;

		xt = solver.step(xt, velocity, t_val, dt);
	}

	return xt; // generated middle frame, flat Float32Array in [-1, 1]
}
```

### Step 6.4 — Benchmark all four targets

```python
# benchmark.py
import time, torch, numpy as np
import onnxruntime as ort
from model import UNet
from inference import generate

N_STEPS = 10
N_RUNS  = 100
SHAPE   = (1, 3, 128, 128)

def measure(fn, name):
    # Warmup
    for _ in range(5): fn()
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(f"{name:<30s}  P50={times[N_RUNS//2]:.1f}ms  P95={times[int(N_RUNS*0.95)]:.1f}ms")

# 1. PyTorch baseline
model = UNet().cuda().eval()
x0 = torch.randn(*SHAPE).cuda(); x1 = torch.randn(*SHAPE).cuda()
c  = torch.randn(1, 512).cuda()
measure(lambda: generate(x0, x1, c, model, N=N_STEPS), "PyTorch native")

# 2. ONNX Runtime (Python, CPU)
sess = ort.InferenceSession("bridge_unet.onnx", providers=["CPUExecutionProvider"])
def onnx_run():
    xt = np.random.randn(*SHAPE).astype(np.float32)
    for i in range(N_STEPS):
        t_v = np.array([(i+0.5)/N_STEPS], dtype=np.float32)
        sess.run(None, {"xt": xt, "t": t_v,
                        "x0": xt.copy(), "c": np.random.randn(1,512).astype(np.float32)})
measure(onnx_run, "ONNX Runtime (CPU)")
```

---

## Final Project Structure

```
schrodinger_bridge/
├── bridge.py           # Brownian Bridge + velocity target
├── model.py            # UNet architecture
├── dataset.py          # Vimeo-90K DataLoader
├── losses.py           # GuidanceLoss (soft constraint)
├── train.py            # Training loop
├── inference.py        # Euler-Maruyama SDE solver
├── eval.py             # PSNR / SSIM / SWD metrics
├── export_onnx.py      # ONNX export
├── benchmark.py        # Speed comparison
├── rust_solver/        # Rust/WASM Euler-Maruyama
│   ├── Cargo.toml
│   └── src/lib.rs
├── web/
│   ├── bridge_inference.js
│   └── index.html
├── checkpoints/
└── data/
    └── vimeo_triplet/
```

---

## Common Bugs and Fixes

| Bug                                        | Symptom                 | Fix                                                                           |
| ------------------------------------------ | ----------------------- | ----------------------------------------------------------------------------- |
| `t` not clamped                            | NaN loss near step 0    | Clamp `t` to `[0.01, 0.99]` in both bridge and velocity functions             |
| Wrong normalisation                        | Blurry/grey outputs     | Confirm `Normalize([0.5]*3, [0.5]*3)` maps to `[-1,1]`, not `[0,1]`           |
| λ too high too early                       | Loss spikes, diverges   | Keep λ=0 for first 20% of steps                                               |
| `torch.compile` left on during ONNX export | Export error            | Use a fresh `UNet()` instance, load weights, then export                      |
| CLIP encoder on wrong device               | CUDA device mismatch    | Pass `device` to `GuidanceLoss.__init__` and call `clip_model.to(device)`     |
| `skip` in ResBlock wrong shape             | RuntimeError in decoder | Ensure `in_ch` passed to ResBlock matches the concatenated skip channel count |
