"""
=============================================================================
  EVALUATION SCRIPT — All 6 Metrics for Schrödinger Bridge Video Editing
=============================================================================
  Computes: CLIP Score, CLIPSIM, LPIPS, SSIM, PSNR, Temporal Warping Error
  
  Designed for: 30 edited videos × 64 frames each
  Hardware:     Runs on CPU (GPU optional, just speeds up CLIP & LPIPS)
  Time:         ~10-30 minutes on a laptop
=============================================================================
"""

import os
import cv2
import csv
import json
import torch
import lpips
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from transformers import CLIPModel, CLIPProcessor


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle numpy float32/float64 for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ==========================================================================
#  CONFIGURATION — Adjust these paths to match your setup
# ==========================================================================
SOURCE_VIDEOS_DIR = "../videos/"            # Original source videos
EDITED_VIDEOS_DIR = "../edited_videos/"      # Output from step6_edit_inference.py
METADATA_FILE     = "../webvid_metadata.csv"
RESULTS_FILE      = "evaluation_results.json"
RESULTS_CSV       = "evaluation_results.csv"
MAX_FRAMES        = 64                      # Frames per video
IMG_SIZE          = 512                     # Must match your pipeline

# Device — will use GPU if available, CPU otherwise (both work fine)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("NOTE: All metrics run fine on CPU. GPU just speeds up CLIP & LPIPS.\n")


# ==========================================================================
#  HELPER FUNCTIONS
# ==========================================================================
def load_video_frames(video_path, max_frames=MAX_FRAMES, size=IMG_SIZE):
    """Load video frames as list of numpy arrays (H, W, 3) in RGB, [0, 255]."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (size, size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def frames_to_tensor(frames):
    """Convert list of numpy frames to torch tensor (N, 3, H, W) in [-1, 1]."""
    arr = np.array(frames).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0  # Scale to [-1, 1] for LPIPS
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)  # (N, 3, H, W)
    return tensor


def frames_to_tensor_01(frames):
    """Convert list of numpy frames to torch tensor (N, 3, H, W) in [0, 1]."""
    arr = np.array(frames).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)  # (N, 3, H, W)
    return tensor


# ==========================================================================
#  METRIC 1: CLIP Score (text-video alignment)
# ==========================================================================
def compute_clip_score(frames, prompt, clip_model, clip_processor):
    """
    CLIP Score = average cosine similarity between text prompt and each frame.
    Higher is better. Measures how well the edit matches the target prompt.
    """
    scores = []
    
    # Encode text once
    text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Process frames in batches to save memory
    batch_size = 16
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        pil_images = [Image.fromarray(f) for f in batch_frames]
        
        image_inputs = clip_processor(images=pil_images, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity
        similarities = (image_features @ text_features.T).squeeze(-1)
        scores.extend(similarities.cpu().numpy().tolist())
    
    return np.mean(scores)


# ==========================================================================
#  METRIC 2: CLIPSIM (temporal consistency via CLIP)
# ==========================================================================
def compute_clipsim(frames, clip_model, clip_processor):
    """
    CLIPSIM = average cosine similarity between CLIP embeddings of 
    consecutive frames. Higher = more temporally consistent.
    """
    # Get all frame embeddings
    all_features = []
    batch_size = 16
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        pil_images = [Image.fromarray(f) for f in batch_frames]
        
        image_inputs = clip_processor(images=pil_images, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        all_features.append(image_features.cpu())
    
    all_features = torch.cat(all_features, dim=0)  # (N, D)
    
    # Compute consecutive frame similarities
    similarities = []
    for i in range(len(all_features) - 1):
        sim = torch.cosine_similarity(
            all_features[i].unsqueeze(0), 
            all_features[i+1].unsqueeze(0)
        ).item()
        similarities.append(sim)
    
    return np.mean(similarities)


# ==========================================================================
#  METRIC 3: LPIPS (perceptual distance, source vs edited)
# ==========================================================================
def compute_lpips(source_frames, edited_frames, lpips_model):
    """
    LPIPS = Learned Perceptual Image Patch Similarity.
    Lower is better (frames are perceptually closer to source).
    Measures structure preservation.
    """
    n = min(len(source_frames), len(edited_frames))
    scores = []
    
    source_tensor = frames_to_tensor(source_frames[:n])  # (N, 3, H, W), [-1, 1]
    edited_tensor = frames_to_tensor(edited_frames[:n])
    
    batch_size = 8
    for i in range(0, n, batch_size):
        src_batch = source_tensor[i:i+batch_size].to(device)
        edt_batch = edited_tensor[i:i+batch_size].to(device)
        
        with torch.no_grad():
            d = lpips_model(src_batch, edt_batch)
        
        scores.extend(d.squeeze().cpu().numpy().tolist() if d.dim() > 1 
                       else [d.item()])
    
    return np.mean(scores)


# ==========================================================================
#  METRIC 4: SSIM (structural similarity, source vs edited)
# ==========================================================================
def compute_ssim(source_frames, edited_frames):
    """
    SSIM = Structural Similarity Index. Range [0, 1].
    Higher = more structurally similar to source. 
    Measures structure preservation.
    """
    n = min(len(source_frames), len(edited_frames))
    scores = []
    
    for i in range(n):
        src = source_frames[i]
        edt = edited_frames[i]
        
        # Resize if needed
        if src.shape != edt.shape:
            edt = cv2.resize(edt, (src.shape[1], src.shape[0]))
        
        score = compare_ssim(src, edt, channel_axis=2, data_range=255)
        scores.append(score)
    
    return np.mean(scores)


# ==========================================================================
#  METRIC 5: PSNR (peak signal-to-noise ratio, source vs edited)
# ==========================================================================
def compute_psnr(source_frames, edited_frames):
    """
    PSNR = Peak Signal-to-Noise Ratio (dB).
    Higher = less distortion from source.
    """
    n = min(len(source_frames), len(edited_frames))
    scores = []
    
    for i in range(n):
        src = source_frames[i]
        edt = edited_frames[i]
        
        if src.shape != edt.shape:
            edt = cv2.resize(edt, (src.shape[1], src.shape[0]))
        
        score = compare_psnr(src, edt, data_range=255)
        scores.append(score)
    
    return np.mean(scores)


# ==========================================================================
#  METRIC 6: Temporal Warping Error
# ==========================================================================
def compute_temporal_warping_error(frames):
    """
    Temporal Warping Error using optical flow (Farneback method).
    Warps frame_i using flow to frame_{i+1}, computes MSE between
    warped frame and actual frame_{i+1}.
    Lower = smoother temporal transitions = better consistency.
    
    Uses OpenCV Farneback — runs on CPU, no GPU needed.
    """
    errors = []
    
    for i in range(len(frames) - 1):
        # Convert to grayscale for optical flow
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
        
        # Compute dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, 
            None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Warp frame_i towards frame_{i+1} using the flow
        h, w = gray1.shape
        map_x = np.float32(np.tile(np.arange(w), (h, 1))) + flow[:, :, 0]
        map_y = np.float32(np.tile(np.arange(h).reshape(-1, 1), (1, w))) + flow[:, :, 1]
        
        warped = cv2.remap(
            frames[i], map_x, map_y, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REFLECT
        )
        
        # MSE between warped frame and actual next frame
        mse = np.mean((warped.astype(np.float32) - frames[i+1].astype(np.float32)) ** 2)
        errors.append(mse)
    
    return np.mean(errors)


# ==========================================================================
#  MAIN EVALUATION PIPELINE
# ==========================================================================
def main():
    print("=" * 70)
    print("  EVALUATION: Schrödinger Bridge Video Editing")
    print("  Metrics: CLIP Score, CLIPSIM, LPIPS, SSIM, PSNR, Warp Error")
    print("=" * 70)
    
    # ------------------------------------------------------------------
    #  Step 1: Load models (one-time cost)
    # ------------------------------------------------------------------
    print("\n[1/3] Loading evaluation models...")
    
    print("  Loading CLIP model (ViT-B/32)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    
    print("  Loading LPIPS model (AlexNet)...")
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    
    print("  ✓ All models loaded.\n")
    
    # ------------------------------------------------------------------
    #  Step 2: Load metadata for prompts
    # ------------------------------------------------------------------
    print("[2/3] Loading metadata...")
    df = pd.read_csv(METADATA_FILE)
    
    # Build prompt lookup from metadata
    # step6 uses: prompt = f"{original_desc}, {art_style}"
    # We'll reconstruct this
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
    
    # ------------------------------------------------------------------
    #  Step 3: Evaluate each edited video
    # ------------------------------------------------------------------
    print("[3/3] Evaluating videos...\n")
    
    # Find all edited videos
    edited_files = sorted([
        f for f in os.listdir(EDITED_VIDEOS_DIR) 
        if f.endswith('.mp4') and '_edited' in f
    ])
    
    if not edited_files:
        print(f"ERROR: No edited videos found in {EDITED_VIDEOS_DIR}")
        print("  Expected filenames like: 12345_edited.mp4")
        print("  Run step6_edit_inference.py first.")
        return
    
    print(f"  Found {len(edited_files)} edited videos.\n")
    
    # Storage for all results
    all_results = []
    
    # Per-metric accumulators
    all_clip_scores = []
    all_clipsim_scores = []
    all_lpips_scores = []
    all_ssim_scores = []
    all_psnr_scores = []
    all_warp_errors = []
    
    for file_idx, edited_file in enumerate(tqdm(edited_files, desc="Evaluating")):
        # Extract video ID
        vid_id = edited_file.replace("_edited.mp4", "")
        
        # Paths
        edited_path = os.path.join(EDITED_VIDEOS_DIR, edited_file)
        source_path = os.path.join(SOURCE_VIDEOS_DIR, f"{vid_id}.mp4")
        
        # Load edited frames
        edited_frames = load_video_frames(edited_path)
        if len(edited_frames) < 2:
            print(f"  Skipping {vid_id}: too few edited frames ({len(edited_frames)})")
            continue
        
        # Get prompt for this video
        row = df[df['videoid'].astype(str) == vid_id]
        if len(row) == 0:
            prompt = "a video"  # fallback
        else:
            original_desc = str(row.iloc[0].get('name', ''))
            art_style = STYLES[file_idx % len(STYLES)]
            prompt = f"{original_desc}, {art_style}"
        
        result = {"video_id": vid_id, "prompt": prompt[:80] + "..."}
        
        # --- CLIP Score ---
        clip_score = compute_clip_score(edited_frames, prompt, clip_model, clip_processor)
        result["clip_score"] = round(clip_score, 4)
        all_clip_scores.append(clip_score)
        
        # --- CLIPSIM (temporal consistency) ---
        clipsim = compute_clipsim(edited_frames, clip_model, clip_processor)
        result["clipsim"] = round(clipsim, 4)
        all_clipsim_scores.append(clipsim)
        
        # --- Temporal Warping Error ---
        warp_err = compute_temporal_warping_error(edited_frames)
        result["warp_error"] = round(warp_err, 2)
        all_warp_errors.append(warp_err)
        
        # --- Source-dependent metrics (LPIPS, SSIM, PSNR) ---
        if os.path.exists(source_path):
            source_frames = load_video_frames(source_path)
            
            if len(source_frames) >= 2:
                # LPIPS
                lpips_score = compute_lpips(source_frames, edited_frames, lpips_model)
                result["lpips"] = round(lpips_score, 4)
                all_lpips_scores.append(lpips_score)
                
                # SSIM
                ssim_score = compute_ssim(source_frames, edited_frames)
                result["ssim"] = round(ssim_score, 4)
                all_ssim_scores.append(ssim_score)
                
                # PSNR
                psnr_score = compute_psnr(source_frames, edited_frames)
                result["psnr"] = round(psnr_score, 2)
                all_psnr_scores.append(psnr_score)
            else:
                result["lpips"] = "N/A"
                result["ssim"] = "N/A"
                result["psnr"] = "N/A"
        else:
            result["lpips"] = "N/A (no source)"
            result["ssim"] = "N/A (no source)"
            result["psnr"] = "N/A (no source)"
        
        all_results.append(result)
    
    # ------------------------------------------------------------------
    #  Print Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Videos evaluated: {len(all_results)}")
    print(f"  Frames per video: up to {MAX_FRAMES}")
    print("-" * 70)
    
    def fmt(values, higher_better=True):
        if not values:
            return "N/A"
        arrow = "↑" if higher_better else "↓"
        return f"{np.mean(values):.4f} ± {np.std(values):.4f} {arrow}"
    
    print(f"  CLIP Score  (text alignment):   {fmt(all_clip_scores, True)}")
    print(f"  CLIPSIM     (temporal consist):  {fmt(all_clipsim_scores, True)}")
    print(f"  LPIPS       (perceptual dist):   {fmt(all_lpips_scores, False)}")
    print(f"  SSIM        (structural sim):    {fmt(all_ssim_scores, True)}")
    print(f"  PSNR (dB)   (signal quality):    {fmt(all_psnr_scores, True)}")
    print(f"  Warp Error  (temporal smooth):   {fmt(all_warp_errors, False)}")
    print("-" * 70)
    
    # ------------------------------------------------------------------
    #  Save results
    # ------------------------------------------------------------------
    # JSON (full details)
    summary = {
        "num_videos": len(all_results),
        "max_frames_per_video": MAX_FRAMES,
        "aggregate_metrics": {
            "clip_score_mean": round(np.mean(all_clip_scores), 4) if all_clip_scores else None,
            "clip_score_std": round(np.std(all_clip_scores), 4) if all_clip_scores else None,
            "clipsim_mean": round(np.mean(all_clipsim_scores), 4) if all_clipsim_scores else None,
            "clipsim_std": round(np.std(all_clipsim_scores), 4) if all_clipsim_scores else None,
            "lpips_mean": round(np.mean(all_lpips_scores), 4) if all_lpips_scores else None,
            "lpips_std": round(np.std(all_lpips_scores), 4) if all_lpips_scores else None,
            "ssim_mean": round(np.mean(all_ssim_scores), 4) if all_ssim_scores else None,
            "ssim_std": round(np.std(all_ssim_scores), 4) if all_ssim_scores else None,
            "psnr_mean": round(np.mean(all_psnr_scores), 2) if all_psnr_scores else None,
            "psnr_std": round(np.std(all_psnr_scores), 2) if all_psnr_scores else None,
            "warp_error_mean": round(np.mean(all_warp_errors), 2) if all_warp_errors else None,
            "warp_error_std": round(np.std(all_warp_errors), 2) if all_warp_errors else None,
        },
        "per_video_results": all_results
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Full results saved to: {RESULTS_FILE}")
    
    # CSV (for easy LaTeX table generation)
    if all_results:
        csv_rows = []
        for r in all_results:
            csv_rows.append({
                "video_id": r["video_id"],
                "clip_score": r.get("clip_score", ""),
                "clipsim": r.get("clipsim", ""),
                "lpips": r.get("lpips", ""),
                "ssim": r.get("ssim", ""),
                "psnr": r.get("psnr", ""),
                "warp_error": r.get("warp_error", ""),
            })
        
        pd.DataFrame(csv_rows).to_csv(RESULTS_CSV, index=False)
        print(f"  CSV results saved to:  {RESULTS_CSV}")
    
    print("\n  Done! Copy the aggregate metrics into your IEEE Access paper.")
    print("=" * 70)


if __name__ == "__main__":
    main()
