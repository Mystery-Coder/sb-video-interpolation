import torch
import cv2
import numpy as np
import os
import glob
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from step2_network import VideoUNet
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 50

def decode_latents(latents, vae):
    with torch.no_grad():
        # Scale back from latent space
        latents = latents / vae.config.scaling_factor
        B, F, C, H, W = latents.shape
        # Flatten batch and frames to pass to VAE
        latents_in = latents.view(B*F, C, H, W)
        
        image = vae.decode(latents_in).sample
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    
    # Reshape back to B, F
    image = image.reshape(B, F, image.shape[1], image.shape[2], 3)
    return image

def main():
    prompt = "A beautiful robot playing on the ground, high quality, cyberpunk" # Custom editing prompt
    
    print("Loading Models...")
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to(device)
    
    model = VideoUNet().to(device)
    
    # Only load strict=False because we only trained the temporal layers
    try:
        model.load_state_dict(torch.load("video_unet_epoch_50.pth"), strict=False)
        print("Loaded Checkpoint Successfully.")
    except Exception as e:
        print(f"Warning: Could not load checkpoint. Using untrained temporal weights. {e}")
        
    model.eval()
    
    # Encode Prompt
    tokens = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_emb = text_encoder(tokens.input_ids.to(device))[0]
    
    # --- VIDEO TO VIDEO EDITING ---
    # Since locally we only trained for a few minutes, the model cannot generate video from pure scratch yet.
    # Instead, we load a real video, scramble it with 80% noise, and let the model denoise it into the new prompt!
    latent_files = glob.glob("../dataset_tensors/*_latents.pt")
    if not latent_files:
        raise ValueError("No video latents found in dataset_tensors!")
    source_latents = torch.load(latent_files[0]).to(device).unsqueeze(0) # Add batch dimension
    print(f"Editing Source Video: {latent_files[0]}")
    
    START_T = 0.8 # Start at 80% noise (retains 20% of the original video's structure!)
    noise = torch.randn_like(source_latents)
    x = (1 - START_T) * source_latents + START_T * noise
    
    dt = START_T / STEPS
    
    print("Running Euler-Maruyama Solver backwards...")
    # Because solver runs backward in standard generation
    for i in tqdm(range(STEPS)):
        t_val = START_T - (i / STEPS) * START_T
        
        t = torch.tensor([t_val], device=device)
        t_sd = (t * 1000).long()
        
        with torch.no_grad():
            drift = model(x, t_sd, text_emb)
        
        noise = torch.randn_like(x)
        
        # SB backward integration (omit noise on final step for sharpest output!)
        if i == STEPS - 1:
            x = x - drift * dt
        else:
            x = x - drift * dt + 0.1 * (dt ** 0.5) * noise
        
    print("Decoding to video...")
    frames = decode_latents(x, vae)[0] # Extract the first batch (16 frames)
    
    # Save video
    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (frames.shape[2], frames.shape[1]))
    for frame in frames:
        # Convert RGB back to BGR for OpenCV
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print("Video saved to output.mp4!")

if __name__ == "__main__":
    main()
