import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models.transformer import (
    TransformerEncoder, 
    Linear
)

from . import vision, hallucinate

class TransformerVisLangDiscEncoder(nn.Module):
    """
    Vision-language encoder. Either extracts feature from visual inputs (images or videos), 
    or hallucinates visual feature from language inputs (text).
    """
    def __init__(self, args, src_dict, embed_tokens):
        self.args = args
        super().__init__()
        self.encoder_v, self.proj_v = self.build_vis_encoder(args)
        self.encoder_l, self.proj_l = self.build_lang_encoder(args, src_dict, embed_tokens, self.encoder_v)
        self.vislang_embed_norm = args.vislang_embed_norm
        self.mmt_inference = args.mmt_inference
    
    def build_vis_encoder(self, args):
        model = vision.__dict__[args.vis_encoder_arch](
            model_path=args.vis_encoder_model_path,
            config_path=args.vis_encoder_config_path,
            codebook=args.vis_encoder_use_codebook,
        )
        if not args.vis_encoder_finetune:
            for param in model.parameters():
                param.requires_grad = False
        if args.vis_encoder_use_codebook:
            proj = Linear(args.vis_encoder_embed_dim, args.vislang_embed_dim, bias=False)
        else:
            proj = None
        return model, proj
    
    def build_lang_encoder(self, args, src_dict, embed_tokens, vis_encoder):
        if args.vis_encoder_hallucinate == 'transformer':
            model = TransformerEncoder(args, src_dict, embed_tokens)
            proj = Linear(args.encoder_embed_dim, args.vislang_embed_dim, bias=False)
        elif args.vis_encoder_hallucinate == 'dalle':
            halluc_args = json.loads(args.halluc_args)
            model = hallucinate.dalle_transformer(vae=vis_encoder.vae,
                                                  src_dict=src_dict,
                                                  model_path=args.halluc_model_path,
                                                  **halluc_args)
            proj = None
        else:
            model = proj = None
        return model, proj
    
    def normalize(self, feat, mode):
        if mode == 'none':
            pass
        elif mode == 'l1':
            feat = F.normalize(feat, p=1, dim=-1)
        elif mode == 'l2':
            feat = F.normalize(feat, p=2, dim=-1)
        else:
            raise Exception('normalization mode not supported:', mode)
        return feat
    
    def forward(self, vis_input, src_tokens, src_lengths):
        ret_v = vis_input is not None
        ret_l = (self.encoder_l is not None) and (self.training or not self.mmt_inference)
        assert ret_v or ret_l, 'Neither visual nor language features are available'

        # extract features from visual input
        if ret_v:
            emb_v = self.encoder_v(vis_input)
            if len(emb_v.shape) == 4:
                emb_v = emb_v.permute(0, 2, 3, 1)  # NCHW -> NHWC

            # if using pretrained codebook, project to target dimension and normalize
            if self.proj_v is not None:
                emb_v = self.proj_v(emb_v)
                emb_v = self.normalize(emb_v, self.vislang_embed_norm)
        else:
            emb_v = None

        # hallucinate visual features from text input
        if ret_l:
            if self.training:
                emb_l, loss = self.encoder_l(src_tokens, emb_v)  # single forward pass at training time
            else:
                emb_l, loss = self.encoder_l(src_tokens)  # autoregressive decoding at test time
            if self.proj_l is not None:
                if isinstance(emb_l, list):
                    emb_l = [self.proj_l(x) for x in emb_l]
                    emb_l = [self.normalize(x, self.vislang_embed_norm) for x in emb_l]
                else:
                    emb_l = self.proj_l(emb_l)
                    emb_l = self.normalize(emb_l, self.vislang_embed_norm)
        else:
            emb_l = loss = None
        
        return {
            "emb_v": emb_v, 
            "emb_l": emb_l,
            "loss": loss
        }