'''
Hallucination transformer by DALL-E
Code adapted from
- https://github.com/lucidrains/DALLE-pytorch
- https://github.com/openai/DALL-E
'''

import torch
import torch.nn as nn
from torch.nn.functional import gumbel_softmax, normalize
from .dalle import DALLE

class DALLEHalluc(nn.Module):
    def __init__(self, model, src_dict, online, gumbel_hard, temp, temp_decay, n_img):
        super().__init__()
        self.model = model
        self.src_dict = src_dict
        self.online = online
        self.gumbel_hard = gumbel_hard
        self.init_temp = temp
        self.temp = temp
        self.temp_decay = temp_decay
        self.n_img = n_img
    
    def set_num_updates(self, num_updates):
        if self.temp_decay > 0:
            self.temp = self.init_temp * (1 - self.temp_decay) ** num_updates
        return self.temp
    
    def _sample(self, logits, gumbel=False, gumbel_hard=False):
        if gumbel:
            logits = logits[:, :, self.model.num_text_tokens:]
            seq = gumbel_softmax(logits, tau=self.temp, dim=-1, hard=gumbel_hard)
        else:
            seq = torch.argmax(logits, dim=-1) - self.model.num_text_tokens
        return seq
        
    def forward(self, src_tokens, img_tokens=None):
        self.model.eval()
        with torch.set_grad_enabled(self.online):
            x = src_tokens
            if img_tokens is not None:
                logits, loss = self.model(text=x, image=img_tokens, return_loss=self.training)
                logits = logits.transpose(1, 2)  # b c n -> b n c
                if self.n_img > 1:
                    img_seq = [self._sample(logits, gumbel=self.online, gumbel_hard=self.gumbel_hard) for _ in range(self.n_img)]
                else:
                    img_seq = self._sample(logits, gumbel=self.online, gumbel_hard=self.gumbel_hard)
            else:
                if self.n_img > 1:
                    img_seq = [self.model.generate_images(text=x, ret_seq=True) for _ in range(self.n_img)]
                else:
                    img_seq = self.model.generate_images(text=x, ret_seq=True)
                loss = None
        return img_seq, loss


def dalle_transformer(vae, src_dict, model_path=None, 
                      online=True, gumbel_hard=False, 
                      temp=5.0, temp_decay=5e-4, n_img=1):
    ckp = torch.load(model_path, map_location='cpu')
    model = DALLE(vae=vae, **ckp['hparams'])
    if model_path is not None:
        model.load_state_dict(ckp['weights'])
    return DALLEHalluc(model, src_dict, online, gumbel_hard, temp, temp_decay, n_img)