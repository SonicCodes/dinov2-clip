import torch

import torch
from torch import nn
import clip
from transformers import AutoModel, AutoImageProcessor

import torch
import clip
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import webdataset as wds
from torch.utils.data import DataLoader
import tqdm
import os
import signal
import sys

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        return x + self.layers(x)

class ProjectorWithResiduals(nn.Module):
    def __init__(self, input_dim=768, output_dim=512, hidden_dim=2048, num_residual_blocks=6):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.residual_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)])
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.hidden_norm(x)
        x = self.output_proj(x)
        x = self.layer_norm(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
class DinoV2CLIP(torch.nn.Module):
    def __init__(self):
        super(DinoV2CLIP, self).__init__()
        self.clip_model, _ = clip.load("ViT-B/16", device=device)
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
        self.projection = ProjectorWithResiduals(input_dim=768, output_dim=512)
        self.projection.load_state_dict(torch.load("[[[ALIGNMENT MODEL PATH]]]"))
        self.projection.to(device)

    def encode_image(self, pixel_values):
        out= self.dino_model(pixel_values=pixel_values)
        out = self.projection(out.last_hidden_state.mean(dim=1))
        return out

    def encode_text(self, text):
        return self.clip_model.encode_text(text)