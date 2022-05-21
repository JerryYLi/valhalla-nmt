'''
Discrete visual encoder by VQGAN VAE
Code adapted from
- https://github.com/lucidrains/DALLE-pytorch/
- https://github.com/CompVis/taming-transformers/
'''

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vqgan_vae import VQGanVAE


class VQGanVAEEmbed(nn.Module):
    def __init__(self, vae, codebook):
        super().__init__()
        self.vae = vae
        self.codebook = codebook

    def forward(self, x):
        seq = self.vae.get_codebook_indices(x)
        if not self.codebook:
            return seq
        idx = F.one_hot(seq, num_classes=self.vae.num_tokens).float()
        embed = idx @ self.vae.model.quantize.embedding.weight
        b, n, d = embed.shape
        h = w = int(math.sqrt(n))
        embed = embed.view(b, h, w, d).permute(0, 3, 1, 2)
        return embed


def vqgan(model_path=None, config_path=None, codebook=False, **kwargs):
    '''
    Combines encoding + quantization steps of discrete VAE
    '''
    model = VQGanVAE(
        vqgan_model_path=model_path,
        vqgan_config_path=config_path
    )
    return VQGanVAEEmbed(model, codebook=codebook)