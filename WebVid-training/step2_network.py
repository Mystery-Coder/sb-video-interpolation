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
        # It takes the (Batch, Channels, Frames, Height, Width) and smooths it.
        self.temporal_layers = nn.Sequential(
            # Frame-to-frame mixing (kernel: time=3, height=1, width=1)
            nn.Conv3d(4, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.SiLU(),
            # Slightly mix spatial and time together
            nn.Conv3d(32, 4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        )
        
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
            out_2d = self.unet(x_in, t_in, text_in).sample
            
        # Reshape for Temporal Block: (Batch, Channels, Frames, H, W)
        out_3d_in = out_2d.view(B, F, C, H, W).permute(0, 2, 1, 3, 4)
        
        # Learn the video motion properties
        out_3d = self.temporal_layers(out_3d_in)
        
        # Reshape back to expected output: (Batch, Frames, Channels, H, W)
        final_out = out_3d.permute(0, 2, 1, 3, 4)
        
        return final_out
