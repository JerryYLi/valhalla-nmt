# Dataset Preparation

## Multi30K

- Tokenized Multi30K dataset is available under `data/multi30k`, credit to https://github.com/LividWo/Revisit-MMT
- Binarize translation data for fairseq
  ```sh
  bash scripts/multi30k/preproc.sh
  ```
- Download image datasets [Flickr30K](http://shannon.cs.illinois.edu/DenotationGraph/) and [MS COCO](https://cocodataset.org/#download), then create symbolic link
  ```sh
  ln -s /path/to/flickr30k data/flickr30k
  ln -s /path/to/mscoco data/mscoco
  ```

## WIT

- Download our compiled WIT translation data from [project page](http://svcl.ucsd.edu/projects/valhalla/). This is a subset of the original [WIT dataset](https://github.com/google-research-datasets/wit/blob/main/DATA.md) with parallel corpora organized for machine translation. The archive also includes tokenized and BPE encoded sentences.
- For each translation task, download images from URLs in `[train|valid|test]_url.txt` to corresponding paths provided in `[train|valid|test]_img.txt`. Image filenames are the MD5 hashes of their URLs.
- Binarize translation data for fairseq
  ```sh
  bash scripts/wit/preproc.sh
  ```