import torch
import cv2
import pandas as pd
import numpy as np
import os
import glob
import logging
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from step2_network import VideoUNet
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, 
    force=True, # Added force=True to override HuggingFace loggers!
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("edit_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 50
START_T = 0.8 

# A diverse list of artistic styles to apply to the videos!
STYLES = [
    "watercolor painting, soft colors, masterpiece",
    "steampunk 3D render, gears and brass, highly detailed",
    "oil painting in the style of Van Gogh, expressive strokes",
    "Pixar style 3D animation, vibrant colors, cute",
    "anime style illustration, Studio Ghibli, beautiful scenery",
    "charcoal sketch, dramatic lighting, highly detailed",
    "origami paper art, colorful paper textures, macro",
    "neon retrowave 1980s aesthetic, glowing vibrant colors",
    "claymation style, stop motion texture, plasticine",
    "cinematic photorealistic, golden hour lighting, 8k resolution",
    "stained glass window art, glowing vibrant light",
    "post-apocalyptic wasteland style, cinematic lighting",
    "fantasy Dungeons and Dragons concept art, magical glowing runes"
]

# Removed hardcoded dictionary to support 50 videos dynamically!

def decode_latents(latents, vae):
    with torch.no_grad():
        latents = latents / vae.config.scaling_factor
        B, F, C, H, W = latents.shape
        latents_in = latents.view(B*F, C, H, W)
        
        # Chunked VAE Decoding to prevent VRAM spikes
        images = []
        chunk_size = 8
        for i in range(0, B*F, chunk_size):
            chunk = latents_in[i:i+chunk_size]
            chunk_img = vae.decode(chunk).sample
            images.append(chunk_img)
            
        image = torch.cat(images, dim=0)
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    image = image.reshape(B, F, image.shape[1], image.shape[2], 3)
    return image

def main():
    logger.info("Loading Models for AI Video Editing...")
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to(device)
    
    model = VideoUNet().to(device)
    
    # Auto-load latest checkpoint
    checkpoints = glob.glob("video_unet_epoch_*.pth")
    if checkpoints:
        latest_ckpt = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        model.load_state_dict(torch.load(latest_ckpt), strict=False)
        logger.info(f"Loaded Checkpoint: {latest_ckpt}")
    else:
        logger.warning("No checkpoint found. Using untrained temporal weights.")
        
    model.eval()
    
    os.makedirs("edited_videos", exist_ok=True)
    
    # Load metadata
    df = pd.read_csv("../webvid_metadata.csv")
    df_subset = df.head(50) # Process first 50 videos
    
    for idx, row in df_subset.iterrows():
        vid_id = str(row['videoid'])
        original_desc = str(row.get('name', ''))
        
        # Cycle through our diverse list of artistic styles
        art_style = STYLES[idx % len(STYLES)]
        
        # Auto-generate the prompt by combining the original layout with the new art style
        prompt = f"{original_desc}, {art_style}"
        
        latent_file = f"../dataset_tensors/{vid_id}_latents.pt"
        if not os.path.exists(latent_file):
            logger.warning(f"Latents not found for {vid_id}, skipping...")
            continue
            
        logger.info(f"--- Editing Video {vid_id} ---")
        logger.info(f"New Custom Prompt: {prompt}")
        
        # Load source latents and add 80% noise (Preserves 20% Structure!)
        source_latents = torch.load(latent_file).to(device).unsqueeze(0)
        noise = torch.randn_like(source_latents)
        x = (1 - START_T) * source_latents + START_T * noise
        
        # Encode Text Prompt
        with torch.no_grad():
            tokens = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_emb = text_encoder(tokens.input_ids.to(device))[0]
            
        dt = START_T / STEPS
        
        # SDE Euler-Maruyama Backward Integration
        for i in tqdm(range(STEPS), desc=f"Denoising {vid_id}"):
            t_val = START_T - (i / STEPS) * START_T
            t = torch.tensor([t_val], device=device)
            t_sd = (t * 1000).long()
            
            with torch.no_grad():
                drift = model(x, t_sd, text_emb)
                
            step_noise = torch.randn_like(x)
            
            # Omit noise injection on final step for sharpness
            if i == STEPS - 1:
                x = x - drift * dt
            else:
                x = x - drift * dt + 0.1 * (dt ** 0.5) * step_noise
                
        logger.info(f"Decoding {vid_id} to pixel space...")
        frames = decode_latents(x, vae)[0]
        
        out_path = f"edited_videos/{vid_id}_edited.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frames.shape[2], frames.shape[1]))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        logger.info(f"Saved AI Edit: {out_path}\n")

if __name__ == "__main__":
    main()
