# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoder
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from fairseq.modules import transformer_layer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from fairseq.models.transformer import (
    TransformerConfig,
)


# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == 'TransformerJointEncoderBase':
        return 'TransformerJointEncoder'
    else:
        return module_name


class PositionalEmbedding2D(nn.Module):
    '''
    2D positional embedding for image grid features.
    num_rows: Number of rows
    num_cols: Number of columns (=num_rows by default)
    embedding_dim: Embedding dimension
    learned: Use learned positional embeddings instead of sinusoidals
    use_2d: Embed row and column indices separately
        - 'sum': add row and column embeddings, each with dim = embedding_dim 
        - 'concat': concatenate row and column embeddings, each with dim = embedding_dim / 2
        - 'none': use 1d positional embedding with flattened features
    '''
    def __init__(self,
        num_rows: int,
        num_cols: int,
        embedding_dim: int,
        padding_idx: int,
        learned: bool = False,
        use_2d: float = 'sum'
    ):
        super().__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.seq_len = self.num_rows * self.num_cols
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.learned = learned
        self.use_2d = use_2d
        self._build_pos_encoder()
    
    def _build_pos_encoder(self):
        if self.use_2d in ['sum', 'concat']:
            if self.use_2d == 'concat':
                assert self.embedding_dim % 2 == 0, 'Embedding dim must be even for concatenated embedding'
                pos_embed_dim = self.embedding_dim // 2
            else:
                pos_embed_dim = self.embedding_dim
            self.embed_row = PositionalEmbedding(
                self.num_rows,
                pos_embed_dim,
                self.padding_idx,
                learned=self.learned,
            )
            self.embed_col = PositionalEmbedding(
                self.num_cols,
                pos_embed_dim,
                self.padding_idx,
                learned=self.learned,
            )
        else:
            self.embed = PositionalEmbedding(
                self.seq_len,
                self.embedding_dim,
                self.padding_idx,
                learned=self.learned,
            )
    
    def forward(
        self,
        input,
        **kwargs
    ):
        # input shape: bs * (h * w) * embed_dim
        assert input.shape[1] == self.seq_len, 'Input shape mismatch'
        assert input.shape[2] == self.embedding_dim, 'Input dimension mismatch'
        bs, seq_len, embed_dim = input.shape
        if self.use_2d in ['sum', 'concat']:
            if self.use_2d == 'concat':
                embed_dim = embed_dim // 2
            dummy_input_row = torch.ones(1, self.num_rows).to(input.device)
            dummy_input_col = torch.ones(1, self.num_cols).to(input.device)
            pos_embed_row = self.embed_row(dummy_input_row, **kwargs)  # 1 x num_rows x embed_dim
            pos_embed_col = self.embed_col(dummy_input_col, **kwargs)  # 1 x num_cols x embed_dim
            pos_embed_row = pos_embed_row.unsqueeze(2).expand(1, self.num_rows, self.num_cols, embed_dim)  # 1 x num_rows x num_cols x embed_dim
            pos_embed_col = pos_embed_col.unsqueeze(1).expand(1, self.num_rows, self.num_cols, embed_dim)  # 1 x num_rows x num_cols x embed_dim
            if self.use_2d == 'sum':
                pos_embed = pos_embed_row + pos_embed_col
            else:
                pos_embed = torch.cat([pos_embed_row, pos_embed_col], -1)
            pos_embed = pos_embed.flatten(1, 2)
        else:
            dummy_input = torch.ones(1, seq_len).to(input.device)
            pos_embed = self.embed(dummy_input, **kwargs)
        return pos_embed


class TransformerJointEncoderBase(FairseqEncoder):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        vis_embed_tokens (torch.nn.Embedding): visual input embedding
    """

    def __init__(self, cfg, args, dictionary, embed_tokens, vis_embed_tokens):
        self.cfg = cfg
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.encoder_layerdrop = cfg.encoder.layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.vis_shape = (args.vis_encoder_grid_h, args.vis_encoder_grid_w)
        self.vis_pool = nn.AdaptiveAvgPool2d(self.vis_shape)
        self.vis_positions = args.vis_encoder_grid_h * args.vis_encoder_grid_w
        self.max_source_positions = cfg.max_source_positions + self.vis_positions

        self.embed_tokens = embed_tokens
        self.vis_embed_tokens = vis_embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                cfg.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.encoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )

        self.vis_embed_positions = (
            PositionalEmbedding2D(
                args.vis_encoder_grid_h,
                args.vis_encoder_grid_w,
                embedding_dim=args.vislang_embed_dim,
                padding_idx=0,
                learned=args.vis_encoder_learned_pos,
                use_2d=args.vis_encoder_use_2d_pos
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(cfg) for i in range(cfg.encoder.layers)]
        )
        self.num_layers = len(self.layers)

        if cfg.encoder.normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, cfg):
        layer = transformer_layer.TransformerEncoderLayerBase(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(
        self,
        src_tokens,
        vis_input,
        token_embedding: Optional[torch.Tensor] = None
    ):
        # embed visual tokens
        def vis_embed(emb_v):
            if self.vis_embed_tokens is not None:
                if len(emb_v.shape) == 2:
                    vis_token_embedding = self.vis_embed_tokens(emb_v)
                else:
                    vis_token_embedding = emb_v @ self.vis_embed_tokens.weight
                emb_v = self.embed_scale * vis_token_embedding
            return emb_v

        # text embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        
        # visual embed tokens
        if isinstance(vis_input, list):
            emb_v = [vis_embed(x) for x in vis_input]
            emb_v = torch.stack(emb_v, dim=0).mean(0)
        else:
            emb_v = vis_embed(vis_input)
        
        # visual positional embeddings
        if len(emb_v.shape) == 4:
            # resize feature maps to self.vis_shape
            if emb_v.shape[1:3] != self.vis_shape:
                emb_v = emb_v.permute(0, 3, 1, 2)  # NHWC -> NCHW
                emb_v = self.vis_pool(emb_v)
                emb_v = emb_v.permute(0, 2, 3, 1)  # NCHW -> NHWC
            # flatten transformer input
            emb_v = emb_v.flatten(1, 2)  # N(HW)C
        if self.vis_embed_positions is not None:
            emb_v = emb_v + self.vis_embed_positions(emb_v)
        
        # vision-language concatenation
        if hasattr(self.args, 'vis_only') and self.args.vis_only:
            x = emb_v
        else:
            x = torch.cat([x, emb_v], 1)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        vis_input,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, vis_input, src_lengths, return_all_hiddens, token_embeddings
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        vis_input,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, vis_input, token_embeddings)
        bs, seqlen = x.shape[:2]
        vis_padding_mask = torch.zeros(bs, seqlen - encoder_embedding.shape[1]).to(encoder_padding_mask.device).bool()
        encoder_padding_mask = torch.cat([encoder_padding_mask, vis_padding_mask], 1)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = src_tokens.ne(self.padding_idx).sum(dim=1, dtype=torch.int32).reshape(-1, 1).contiguous()
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None or self.vis_embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, 
                   self.embed_positions.max_positions + self.vis_embed_positions.seq_len)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerJointEncoder(TransformerJointEncoderBase):
    def __init__(self, args, dictionary, embed_tokens, vis_embed_tokens):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            args,
            dictionary,
            embed_tokens,
            vis_embed_tokens
        )

    def build_encoder_layer(self, args):
        return super().build_encoder_layer(
            TransformerConfig.from_namespace(args),
        )
