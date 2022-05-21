import torch
import torch.nn as nn

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from models import (
    FairseqEncoder,
    FairseqDecoder,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerDecoder,
    Embedding,
)
from fairseq.models.fairseq_model import check_type

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


class FairseqVisLangJointEncoderDecoderModel(BaseFairseqModel):
    """Base class for multi-modal encoder-decoder models.

    Args:
        encoder_vl (FairseqEncoder): the joint vision-language encoder
        encoder_v (nn.Module): the vision encoder
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, encoder_vl, encoder_v, decoder):
        super().__init__()

        self.encoder_vl = encoder_vl
        self.encoder_v = encoder_v
        self.decoder = decoder

        check_type(self.encoder_vl, FairseqEncoder)
        check_type(self.encoder_v, nn.Module)
        check_type(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, vis_input=None, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            vis_input (FloatTensor): visual input, optional

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_v_out = self.encoder_v(vis_input, src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = {}
        
        # multimodal stream using extracted visual features
        halluc_only = hasattr(self.args, 'halluc_only') and self.args.halluc_only
        if (not halluc_only) and encoder_v_out['emb_v'] is not None:
            encoder_vl_out = self.encoder_vl(src_tokens, encoder_v_out['emb_v'], src_lengths=src_lengths, **kwargs)
            decoder_out['vislang'] = self.decoder(
                prev_output_tokens, encoder_out=encoder_vl_out, **kwargs
            )
        else:
            decoder_out['vislang'] = None
        
        # hallucination stream using text features only
        if encoder_v_out['emb_l'] is not None:
            if isinstance(encoder_v_out['emb_l'], list):
                encoder_vl_out = [self.encoder_vl(src_tokens, x, src_lengths=src_lengths, **kwargs) for x in encoder_v_out['emb_l']]
                decoder_out_list = [self.decoder(prev_output_tokens, encoder_out=x, **kwargs) for x in encoder_vl_out]
                decoder_out_list, decoder_out_extra = zip(*decoder_out_list)
                decoder_out_avg = torch.stack(decoder_out_list, dim=0).mean(0)
                decoder_out['halluc'] = (decoder_out_avg, decoder_out_extra)
            else:
                encoder_vl_out = self.encoder_vl(src_tokens, encoder_v_out['emb_l'], src_lengths=src_lengths, **kwargs)
                decoder_out['halluc'] = self.decoder(prev_output_tokens, encoder_out=encoder_vl_out, **kwargs)

            decoder_out['loss'] = None if halluc_only else encoder_v_out['loss']
        else:
            return decoder_out['vislang']
        
        return decoder_out
    
    def forward_encoder(self, src_tokens, src_lengths, prev_output_tokens, vis_input=None, **kwargs):
        encoder_v_out = self.encoder_v(vis_input, src_tokens, src_lengths=src_lengths, **kwargs)
        if isinstance(encoder_v_out, dict):
            # use hallucinated features at inference time
            if 'emb_l' in encoder_v_out and encoder_v_out['emb_l'] is not None:
                encoder_v_out = encoder_v_out['emb_l']
            else:
                encoder_v_out = encoder_v_out['emb_v']
        if hasattr(self.args, 'rand_inference') and self.args.rand_inference:
            encoder_v_out = torch.randint_like(encoder_v_out, self.args.vis_encoder_tokens)
        encoder_vl_out = self.encoder_vl(src_tokens, encoder_v_out, src_lengths=src_lengths, **kwargs)
        return encoder_vl_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, vis_input=None, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_v_out = self.encoder_v(vis_input, src_tokens, src_lengths=src_lengths, **kwargs)
        features = {}
        
        # multimodal stream using extracted visual features
        if encoder_v_out['emb_v'] is not None:
            encoder_vl_out = self.encoder_vl(src_tokens, encoder_v_out['emb_v'], src_lengths=src_lengths, **kwargs)
            features['vislang'] = self.decoder.extract_features(
                prev_output_tokens, encoder_out=encoder_vl_out, **kwargs
            )
        else: 
            features['vislang'] = None
        
        # hallucination stream using text features only
        if encoder_v_out['emb_l'] is not None:
            encoder_vl_out = self.encoder_vl(src_tokens, encoder_v_out['emb_l'], src_lengths=src_lengths, **kwargs)
            features['halluc'] = self.decoder.extract_features(
                prev_output_tokens, encoder_out=encoder_vl_out, **kwargs
            )
            features['loss'] = encoder_v_out['loss']
        else:
            return features['vislang']

        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder_vl.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


@register_model("vldtransformer")
class VisLangDiscTransformerModel(FairseqVisLangJointEncoderDecoderModel):
    """
    Vision-language Transformer model with discrete embeddings.

    Args:
        encoder_vl (FairseqEncoder): the language/text encoder
        encoder_v (nn.Module): the vision encoder
        decoder (FairseqDecoder): the decoder
    """
    def __init__(self, args, encoder_vl, encoder_v, decoder):
        super().__init__(encoder_vl, encoder_v, decoder)
        self.args = args
        self.supports_align_args = True
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--checkpoint-activations', action='store_true',
                            help='checkpoint activations at each layer, which saves GPU '
                                 'memory usage at the cost of some additional compute')
        parser.add_argument('--offload-activations', action='store_true',
                            help='checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations.')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # args for vision encoder
        parser.add_argument('--vis-encoder-arch', type=str, 
                            help='vision encoder architecture')
        parser.add_argument('--vis-encoder-model-path', type=str, 
                            help='path to vision encoder checkpoint')
        parser.add_argument('--vis-encoder-config-path', type=str, 
                            help='path to vision encoder configs')
        parser.add_argument('--vis-encoder-finetune', action='store_true', 
                            help='finetune visual encoder')
        parser.add_argument('--vis-encoder-use-codebook', action='store_true', 
                            help='use pretrained visual codebook')
        parser.add_argument('--vis-encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained visual encoder embedding')
        parser.add_argument('--vis-encoder-embed-dim', type=int, 
                            help='vision encoder feature dimension')
        parser.add_argument('--vis-encoder-tokens', type=int, 
                            help='vision encoder vocabulary size')
        parser.add_argument('--vis-encoder-grid-h', type=int, 
                            help='vision encoder feature grid height')
        parser.add_argument('--vis-encoder-grid-w', type=int, 
                            help='vision encoder feature grid width')
        parser.add_argument('--vis-encoder-learned-pos', action='store_true', 
                            help='use learned positional embedding for visual features')
        parser.add_argument('--vis-encoder-use-2d-pos', type=str, 
                            help='use 2d positional embedding for visual features')
        parser.add_argument('--vis-encoder-hallucinate', type=str, 
                            help='hallucinate vision encoder during training')
        parser.add_argument('--vis-only', action='store_true', 
                            help='use visual input only, ignoring input sentence')
        # args for hallucination
        parser.add_argument('--halluc-model-path', type=str, 
                            help='path to hallucination transformer checkpoint')
        parser.add_argument('--halluc-args', type=str, 
                            help='hallucination transformer args')
        parser.add_argument('--mmt-inference', action='store_true', 
                            help='use ground-truth image embeddings at test time')
        parser.add_argument('--rand-inference', action='store_true', 
                            help='use random image embeddings at test time')
        parser.add_argument('--halluc-only', action='store_true', 
                            help='use hallucination stream only')
        parser.add_argument('--pretrain-mmt', type=str, 
                            help='path to load pretrained MMT transformer')
        # args for vision-language contrastive learning
        parser.add_argument('--vislang-embed-dim', type=int, 
                            help='multimodal embedding dimension')
        parser.add_argument('--vislang-embed-norm', type=str, 
                            help='normalize multimodal embedding (l1/l2/none)')
        # args for Fully Sharded Data Parallel (FSDP) training
        parser.add_argument(
            '--min-params-to-wrap', type=int, metavar='D', default=DEFAULT_MIN_PARAMS_TO_WRAP,
            help=(
                'minimum number of params for a layer to be wrapped with FSDP() when '
                'training with --ddp-backend=fully_sharded. Smaller values will '
                'improve memory efficiency, but may make torch.distributed '
                'communication less efficient due to smaller input sizes. This option '
                'is set to 0 (i.e., always wrap) when --checkpoint-activations or '
                '--offload-activations are passed.'
            )
        )
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if not args.vis_encoder_use_codebook:
            vis_embed_tokens = nn.Embedding(args.vis_encoder_tokens, args.encoder_embed_dim, padding_idx=None)
            nn.init.normal_(vis_embed_tokens.weight, mean=0, std=args.encoder_embed_dim ** -0.5)
        else:
            vis_embed_tokens = None
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder_vl = cls.build_lang_encoder(args, src_dict, encoder_embed_tokens, vis_embed_tokens)
        encoder_v = cls.build_vislang_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder_vl = fsdp_wrap(encoder_vl, min_num_params=min_params_to_wrap)
            encoder_v = fsdp_wrap(encoder_v, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)
        return cls(args, encoder_vl, encoder_v, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_lang_encoder(cls, args, src_dict, embed_tokens, vis_embed_tokens):
        from models.transformer_discrete_encoder import TransformerJointEncoder
        encoder = TransformerJointEncoder(args, src_dict, embed_tokens, vis_embed_tokens)
        if hasattr(args, 'pretrain_mmt') and args.pretrain_mmt is not None:
            pref = 'encoder_vl.'
            ckp = torch.load(args.pretrain_mmt)['model']
            enc_ckp = {k[len(pref):]: v for k, v in ckp.items() if k.startswith(pref)}
            encoder.load_state_dict(enc_ckp)
        return encoder

    @classmethod
    def build_vislang_encoder(cls, args, src_dict, embed_tokens):
        from models.vislang_discrete_encoder import TransformerVisLangDiscEncoder
        encoder = TransformerVisLangDiscEncoder(args, src_dict, embed_tokens)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
        if hasattr(args, 'pretrain_mmt') and args.pretrain_mmt is not None:
            pref = 'decoder.'
            ckp = torch.load(args.pretrain_mmt)['model']
            dec_ckp = {k[len(pref):]: v for k, v in ckp.items() if k.startswith(pref)}
            decoder.load_state_dict(dec_ckp)
        return decoder

@register_model_architecture("vldtransformer", "vldtransformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    # vision encoder args
    args.vis_encoder_arch = getattr(args, "vis_encoder_arch", "vqgan")
    # args.vis_encoder_args = getattr(args, "vis_encoder_args", "{}")
    args.vis_encoder_model_path = getattr(args, "vis_encoder_model_path", None)
    args.vis_encoder_config_path = getattr(args, "vis_encoder_config_path", None)
    args.vis_encoder_finetune = getattr(args, "vis_encoder_finetune", False)
    args.vis_encoder_use_codebook = getattr(args, "vis_encoder_use_codebook", False)
    args.vis_encoder_embed_path = getattr(args, "vis_encoder_embed_path", None)
    args.vis_encoder_embed_dim = getattr(args, "vis_encoder_embed_dim", 128)
    args.vis_encoder_tokens = getattr(args, "vis_encoder_tokens", 8192)
    args.vis_encoder_grid_h = getattr(args, "vis_encoder_grid_h", 4)
    args.vis_encoder_grid_w = getattr(args, "vis_encoder_grid_w", args.vis_encoder_grid_h)
    args.vis_encoder_learned_pos = getattr(args, "vis_encoder_learned_pos", False)
    args.vis_encoder_use_2d_pos = getattr(args, "vis_encoder_use_2d_pos", 'sum')
    args.vis_encoder_hallucinate = getattr(args, "vis_encoder_hallucinate", 'none')
    args.vis_only = getattr(args, "vis_only", False)

    # vision-language learning args
    args.vislang_embed_dim = getattr(args, "vislang_embed_dim", args.encoder_embed_dim)
    args.vislang_embed_norm = getattr(args, "vislang_embed_norm", "none")
    args.halluc_model_path = getattr(args, "halluc_model_path", None)
    args.halluc_args = getattr(args, "halluc_args", "{}")
    args.mmt_inference = getattr(args, "mmt_inference", False)
    args.rand_inference = getattr(args, "rand_inference", False)
    args.halluc_only = getattr(args, "halluc_only", False)
    args.pretrain_mmt = getattr(args, "pretrain_mmt", None)


@register_model_architecture("vldtransformer", "vldtransformer_small")
def small_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    return base_architecture(args)


@register_model_architecture("vldtransformer", "vldtransformer_tiny")
def tiny_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    return base_architecture(args)