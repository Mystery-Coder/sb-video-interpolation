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

# Dictionary mapping Video ID to our Custom Edit Prompts
# These prompts are specifically designed to match the structural geometry of the original videos!
EDIT_PROMPTS = {
    "1016280532": "A cute golden retriever puppy playing on a wooden floor at home, highly detailed",
    "1032401924": "Futuristic cyberpunk hover-buses inside a neon-lit sci-fi transit station, 4k",
    "1022131720": "Portrait of a majestic elven queen in ornate golden armor smiling into camera on black background, fantasy art",
    "1023444685": "A glowing mechanical robotic rabbit in a metallic cyberpunk basket, highly detailed",
    "31313158": "Animation of a walking metallic cyborg robot on a neon wireframe background, 4k",
    "11857364": "Four glowing crystalline shards floating on white, highly detailed 3D render",
    "1065907894": "Excited astronaut in a spacesuit isolated on black background looking at a glowing hologram, sci-fi",
    "1051185190": "A steampunk wizard inspects glowing magical artifacts on shelves in a mystical library, 4k",
    "1062687892": "Aerial of a futuristic sci-fi colony village on the surface of Mars, 4k",
    "1019861101": "A futuristic cyberpunk city skyline at night with a glowing neon blue bridge, flying cars traffic time lapse, 4k",
    "25979636": "Scenery of a river of molten lava in a volcanic wasteland, cinematic lighting",
    "18599999": "A futuristic cyborg girl standing on a skyscraper roof with arms out enjoying the neon city view",
    "1031360726": "A bouquet of glowing crystal flowers blooming. isolated on black background. macro time lapse",
    "15386587": "A heavily armored space marine shooting a glowing laser hole in a spaceship wall, 4k",
    "1027305596": "Flowing liquid gold and silver abstract background animation, highly detailed"
}

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
    
    for vid_id, prompt in EDIT_PROMPTS.items():
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
