import os
import torch
import cv2
import pandas as pd
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
VIDEOS_DIR = "../videos/"
METADATA_FILE = "../metadata.csv"
OUTPUT_DIR = "../dataset_tensors/"
MAX_FRAMES = 16
IMG_SIZE = 256 # Switch to 512 on ROCm instance

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    if len(frames) < MAX_FRAMES:
        # pad with last frame if video is too short
        while len(frames) < MAX_FRAMES:
            frames.append(frames[-1] if frames else torch.zeros((IMG_SIZE, IMG_SIZE, 3)))
            
    # Normalize to [-1, 1] for VAE
    frames_tensor = torch.tensor(frames).float() / 127.5 - 1.0
    frames_tensor = frames_tensor.permute(0, 3, 1, 2) # (F, C, H, W)
    return frames_tensor

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading VAE and CLIP models...")
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to(device)
    
    df = pd.read_csv(METADATA_FILE)
    
    print("Processing videos and caching latents...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        vid_id = str(row['videoid'])
        prompt = str(row.get('name', ''))
        
        vid_path = os.path.join(VIDEOS_DIR, f"{vid_id}.mp4")
        if not os.path.exists(vid_path):
            continue
            
        # 1. Encode Video to Latents
        with torch.no_grad():
            frames_tensor = extract_frames(vid_path).to(device)
            # Encode frames through VAE
            latents = vae.encode(frames_tensor).latent_dist.sample()
            latents = latents * vae.config.scaling_factor # Standard stable diffusion scaling
            
        # 2. Encode Prompt to Text Embeddings
        with torch.no_grad():
            tokens = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(tokens.input_ids.to(device))[0]
            
        # Save to disk
        torch.save(latents.cpu(), os.path.join(OUTPUT_DIR, f"{vid_id}_latents.pt"))
        torch.save(text_embeddings.cpu(), os.path.join(OUTPUT_DIR, f"{vid_id}_text.pt"))

if __name__ == "__main__":
    main()
