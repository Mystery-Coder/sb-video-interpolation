import os
import glob
import torch
import torch.nn as nn
import logging
from torch.utils.data import Dataset, DataLoader
from step2_network import VideoUNet
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "../dataset_tensors/"
BATCH_SIZE = 8 # <-- ROCm MODIFY: Increased to 8 to fully saturate the 192GB MI300X!
EPOCHS = 50
LR = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

class LatentVideoDataset(Dataset):
    def __init__(self, data_dir):
        self.video_files = glob.glob(os.path.join(data_dir, "*_latents.pt"))
        
    def __len__(self):
        return len(self.video_files)
        
    def __getitem__(self, idx):
        latent_path = self.video_files[idx]
        text_path = latent_path.replace("_latents.pt", "_text.pt")
        
        latents = torch.load(latent_path)
        text_emb = torch.load(text_path)
        
        # Text embedding needs squeeze to fix extra dimension
        if len(text_emb.shape) == 3:
            text_emb = text_emb.squeeze(0)
            
        return latents, text_emb

def main():
    logger.info(f"Using device: {device}")
    
    # 1. Setup Model
    model = VideoUNet().to(device)
    
    # Only optimize the temporal layers!
    optimizer = torch.optim.AdamW(model.temporal_layers.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # 2. Setup Data
    dataset = LatentVideoDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    logger.info(f"Starting training on {len(dataset)} videos...")
    
    # 3. Soft-Constrained Schrödinger Bridge Loop
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for latents, text_emb in pbar:
            latents = latents.to(device)
            text_emb = text_emb.to(device)
            B, F, C, H, W = latents.shape
            
            optimizer.zero_grad()
            
            # --- SCHRÖDINGER BRIDGE MATH ---
            # Random time selection t in [0, 1]
            t = torch.rand((B,), device=device)
            
            # Pure Gaussian noise (The generic target distribution for normal diffusion)
            # In a full SB, this would be x1. We treat noise as x1 here.
            noise = torch.randn_like(latents)
            
            # Reshape t for broadcasting
            t_expand = t.view(B, 1, 1, 1, 1)
            sigma = 0.1 # Bridge volatility
            
            # The corrupted path (interpolation)
            x_t = (1 - t_expand) * latents + t_expand * noise + sigma * torch.sqrt(t_expand * (1 - t_expand) + 1e-8) * torch.randn_like(latents)
            
            # The theoretical target velocity (drift)
            # Pushing from current x_t towards the noise target
            target_drift = (noise - latents) + sigma * (1 - 2*t_expand) / (2 * torch.sqrt(t_expand*(1-t_expand) + 1e-8)) * torch.randn_like(latents)
            
            # --- MODEL PREDICTION ---
            # t needs to be mapped to typical 1000 timestep range for SD UNet
            t_sd = (t * 1000).long() 
            
            pred_drift = model(x_t, t_sd, text_emb)
            
            # --- LOSS & OPTIMIZE ---
            loss = criterion(pred_drift, target_drift)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        logger.info(f"Epoch {epoch+1} Average Loss: {total_loss/len(dataloader):.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            torch.save(model.state_dict(), f"video_unet_epoch_{epoch+1}.pth")
            logger.info("Checkpoint saved!")

if __name__ == "__main__":
    main()
