# VALHALLA: Visual Hallucination for Machine Translation 
[[arXiv]](https://arxiv.org/abs/2206.00100) [[Project Page]](http://www.svcl.ucsd.edu/projects/valhalla/)

This repository contains code for CVPR'22 submission "VALHALLA: Visual Hallucination for Machine Translation".

If you use the codes and models from this repo, please cite our work. Thanks!

```
@inproceedings{li2022valhalla,
    title={{VALHALLA: Visual Hallucination for Machine Translation}},
    author={Li, Yi and Panda, Rameswar and Kim, Yoon and Chen, Chun-Fu (Richard) and Feris, Rogerio and Cox, David and Vasconcelos, Nuno},
    booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
```

## Get started

```sh
# Clone repository
git clone --recursive https://github.com/JerryYLi/valhalla-nmt.git

# Install prerequisites
conda env create -f environments/basic.yml  # basic environment for machine translation from pretrained hallucination models; or
conda env create -f environments/full.yml  # complete environment for training visual encoder, hallucination and translation from scratch
conda activate valhalla

# Build environment
pip install -e fairseq/
pip install -e taming-transformers/  # for training vqgan visual encoders
pip install -e DALLE-pytorch/  # for training hallucination transformers
```

## Data

All datasets should be organized under `data/` directory, while binarized translation datasets are created under `data-bin/` by preparation scripts. Refer to [`data/README.md`](data/README.md) for detailed instructions.

## Training

### Stage 1: Discrete visual encoder

Pretrain VQGAN visual encoder using [taming-transformers](https://github.com/CompVis/taming-transformers). We provide a modified version under `taming-transformers/` directory containing all scripts and configs needed for Multi30K and WIT data.

```sh
cd taming-transformers/

# Multi30K
ln -s [FLICKR_ROOT] data/multi30k/images  
python main.py --base configs/multi30k/vqgan_c128_l6.yaml --train True --gpus 0,1,2,3

# WIT
ln -s [WIT_ROOT] data/wit/images
python main.py --base configs/wit/vqgan_c128_l6_[TASK].yaml --train True --gpus 0,1,2,3

cd ..
```

### Stage 2: Hallucination transformer

Pretrain DALL-E hallucination transformer using [DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch). 

```sh
cd DALLE-pytorch/

# Multi30K
python train_dalle.py \
  --data_path ../data-bin/multi30k/[SRC]-[TGT]/ --src_lang [SRC] --tgt_lang [TGT] \
  --vqgan_model_path [VQGAN_CKP].ckpt --vqgan_config_path [VQGAN_CFG].yaml \
  --taming --depth 2 --truncate_captions \
  --vis_data flickr30k --vis_data_dir [FLICKR_ROOT] --text_seq_len 16 --save_name "multi30k_[SRC]_[TGT]"

# WIT
python train_dalle.py \
  --data_path ../data-bin/wit/[SRC]_[TGT]/ --src_lang [SRC] --tgt_lang [TGT] \
  --vqgan_model_path [VQGAN_CKP].ckpt --vqgan_config_path [VQGAN_CFG].yaml \
  --taming --depth 2 --truncate_captions \
  --vis_data wit --vis_data_dir [WIT_ROOT] --text_seq_len 32 --save_name "wit_[SRC]_[TGT]"

cd ..
```

### Stage 3: Multimodal translation

Train VALHALLA model on top of visual encoders from stage 1 and hallucination models from stage 2.
```sh
# Multi30K
bash scripts/multi30k/train.sh -s [SRC] -t [TGT] -w [VIS_CKP] -x [VIS_CFG] -u [HAL_CKP]

# WIT
bash scripts/wit/train.sh -s [SRC] -t [TGT] -b [TOKENS] -w [VIS_CKP] -x [VIS_CFG] -u [HAL_CKP]
```
- Additional options:
  - `-a`: Architecture (choose from `vldtransformer/vldtransformer_small/vldtransformer_tiny`)
  - `-e`: Consistency loss weight (default: 0.5)
  - `-g`: Hallucination loss weight (default: 0.5)

## Testing

### Evaluating VALHALLA models:
```sh
# Multi30K
bash scripts/multi30k/test.sh -s [SRC] -t [TGT]

# WIT
bash scripts/wit/test.sh -s [SRC] -t [TGT] -b [TOKENS]
```
- Additional options:
  - `-g`: Multimodal inference, use ground-truth visual context from test images
  - `-k`: Average last K checkpoints

## Pretrained Models

Pretrained models are available for download at [Releases page](https://github.com/JerryYLi/valhalla-nmt/releases/).

## Acknowledgements

This project is built on several open-source repositories/codebases, including:
- [fairseq](https://github.com/pytorch/fairseq)
- [Revisit-MMT](https://github.com/LividWo/Revisit-MMT)
- [taming-transformers](https://github.com/CompVis/taming-transformers)
- [DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch)
- [subword-nmt](https://github.com/rsennrich/subword-nmt)
- [mosesdecoder](https://github.com/moses-smt/mosesdecoder)
