import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class VideoUNet(nn.Module):
    """
    A simple wrapper that applies the Frozen Stable Diffusion 2D U-Net
    to every frame individually, and then learns Temporal Consistency 
    using a custom 3D Convolution layer at the very end.
    """
    def __init__(self):
        super().__init__()
        
        # Load the massive pre-trained image U-Net
        print("Loading Pre-Trained SD 1.5 U-Net...")
        self.unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        
        # FREEZE the 2D spatial layers so we don't forget what the world looks like!
        for param in self.unet.parameters():
            param.requires_grad = False
            
        # Define a lightweight 3D Temporal module
        # This is the ONLY part of the network that will be trained.
        self.temporal_layers = nn.Sequential(
            # Frame-to-frame mixing (kernel: time=3, height=1, width=1)
            nn.Conv3d(4, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.SiLU(),
            # Slightly mix spatial and time together
            nn.Conv3d(32, 4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )
        
        # CRITICAL FIX: Zero-initialize the last layer!
        # This means the temporal block starts by doing nothing (adding 0),
        # allowing the pre-trained spatial UNet to do the heavy lifting perfectly!
        nn.init.zeros_(self.temporal_layers[-1].weight)
        nn.init.zeros_(self.temporal_layers[-1].bias)
        
    def forward(self, x, t, encoder_hidden_states):
        # x input shape: (Batch, Frames, Channels=4, Height=32, Width=32)
        B, F, C, H, W = x.shape
        
        # Flatten the frames into the Batch dimension so the 2D UNet can process them
        x_in = x.view(B * F, C, H, W)
        
        # Expand timestep and text embeddings to match the flattened batch
        t_in = t.repeat_interleave(F) if len(t.shape) == 1 else t
        text_in = encoder_hidden_states.repeat_interleave(F, dim=0)
        
        # Forward pass through frozen Stable Diffusion
        with torch.no_grad():
            t_real = (t_in / 1000.0).view(-1, 1, 1, 1)
            
            # --- VARIANCE SCALING FIX ---
            # Our SB interpolation: x_t = (1-t)x0 + t*noise has variance (1-t)^2 + t^2
            # But the Stable Diffusion UNet mathematically expects inputs with variance exactly = 1.0!
            # If we don't scale it, the UNet sees a "dim" image and outputs blurry garbage.
            std = torch.sqrt((1.0 - t_real)**2 + t_real**2 + 1e-8)
            x_scaled = x_in / std
            
            # Pass the correctly scaled input to SD
            noise_pred = self.unet(x_scaled, t_in, text_in).sample
            
            # --- CRITICAL MATHEMATICAL FIX ---
            # Stable Diffusion predicts NOISE. But our Schrödinger Bridge expects DRIFT (noise - x0).
            # We can algebraically recover x0: x0 = (x_in - t*noise) / (1-t)
            # Notice we use the original x_in here for recovery!
            x0_pred = (x_in - t_real * noise_pred) / (1.0 - t_real + 1e-5)
            
            # The base structural drift
            base_drift = noise_pred - x0_pred
            
        # Reshape for Temporal Block: (Batch, Channels, Frames, H, W)
        out_3d_in = base_drift.view(B, F, C, H, W).permute(0, 2, 1, 3, 4)
        
        # Learn the video motion properties (Residual)
        out_3d = self.temporal_layers(out_3d_in)
        temporal_out = out_3d.permute(0, 2, 1, 3, 4)
        
        # Add as a residual connection!
        final_out = base_drift.view(B, F, C, H, W) + temporal_out
        
        return final_out
