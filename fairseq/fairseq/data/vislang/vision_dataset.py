import os
import random
from tqdm import tqdm
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms

__all__ = ['ImagePathDataset', 'flickr30k', 'mscoco', 'wit']


class ImagePathDataset(data.Dataset):
    '''
    Plain image dataset

    Args:
        root: root directory of image files
        anno: list of image filenames, or txt file containing the list
        transform: torchvision image transforms
    '''
    def __init__(self, root, anno, transform=None):
        super().__init__()
        self.root = root
        if isinstance(anno, list):
            self.anno = anno
        else:
            with open(anno, 'r') as f:
                self.anno = [l.strip('\n') for l in f]
        self.transform = transform
    
    def __len__(self):
        return len(self.anno)
    
    def check_files(self):
        total = valid = 0
        for line in tqdm(self.anno):
            for fn in line.split(';'):
                total += 1
                fp = os.path.join(self.root, fn)
                if os.path.isfile(fp):
                    valid += 1
        print(f'{valid} images ({total} total)')
    
    def __getitem__(self, idx):
        fn = self.anno[idx]
        if ';' in fn:
            fn = random.choice(fn.split(';'))
        fp = os.path.join(self.root, fn)
        if os.path.getsize(fp) <= 0:
            return random.choice(self)
        img = Image.open(fp).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


def get_transform(train=True, 
                  crop=224, 
                  resize=256, 
                  randresize=False,
                  randflip=True,
                  normalize=False,
                  norm_mean=[0.485, 0.456, 0.406],
                  norm_std=[0.229, 0.224, 0.225]
):
    """
    Build transform for image datasets

    Args:
        train: Use random crop and horizontal flip
        crop: Crop dimension
        resize: Resize dimension
        randresize: Use RandomResizedCrop during training
        randflip: Use RandomHorizontalFlip during training
        normalize: Use normalization
        norm_mean: Channel mean for normalization
        norm_std: Channel standard deviation for normalization
    """
    transform_list = []
    if train:
        if randresize:
            transform_list.append(transforms.RandomResizedCrop(crop))
        else:
            transform_list.append(transforms.Resize(resize))
            transform_list.append(transforms.RandomCrop(crop))
        if randflip:
            transform_list.append(transforms.RandomHorizontalFlip())
    else:
        transform_list.append(transforms.Resize(resize))
        transform_list.append(transforms.CenterCrop(crop))
    transform_list.append(transforms.ToTensor())
    if normalize:
        transform_list.append(transforms.Normalize(norm_mean, norm_std))
    return transforms.Compose(transform_list)


def flickr30k(
    root='data/flickr30k',
    m30k_root='data/multi30k',
    split='train',
    test_year=2016,
    **kwargs
):
    """
    Flickr30K image dataset for Multi30K
    args:
        root: root directory to Flickr30K images
        m30k_root: root directory to Multi30K captions
        split: train | val | test
        test_year: 2016 | 2017 | 2018
    """
    if split == 'test':
        split_fn = f'test_{test_year}_flickr.txt'
        if test_year in [2017, 2018]:
            data_root = os.path.join(root, f'test_{test_year}')
        else:
            data_root = os.path.join(root, 'flickr30k-images')
    else:
        split_fn = 'train.txt' if 'train' in split else 'val.txt'
        data_root = os.path.join(root, 'flickr30k-images')
    
    transform = get_transform(train='train' in split, **kwargs)
    
    return ImagePathDataset(
        root=data_root,
        anno=os.path.join(m30k_root, 'image_splits', split_fn),
        transform=transform
    )


def mscoco(
    root='data/mscoco',
    m30k_root='data/multi30k',
    split='train',
    **kwargs
):
    """
    MS COCO image dataset for Multi30K
    args:
        root: root directory to COCO images
        m30k_root: root directory to Multi30K captions
        split: test only
    """
    assert 'test' in split, 'only test set is supported'
    split_fn = 'test_2017_mscoco.txt'
    transform = get_transform(train=False, **kwargs)
    anno = os.path.join(m30k_root, 'image_splits', split_fn)
    anno_list = []
    with open(anno, 'r') as f:
        for l in f:
            fn = l.strip('\n').split('#')[0]
            split_dir = fn.split('_')[1]
            anno_list.append(os.path.join(split_dir, fn))
    
    return ImagePathDataset(
        root=root,
        anno=anno_list,
        transform=transform
    )


def wit(
    root='data/wit',
    lang='en_de',
    split='train',
    **kwargs
):
    """
    WIT image dataset
    args:
        root: root directory to WIT images
        split: train | val | test
    """
    split_fn = f'mmt/{lang}/{split}_img.txt'
    transform = get_transform(train='train' in split, **kwargs)
    anno = os.path.join(root, split_fn)
    
    return ImagePathDataset(
        root=root,
        anno=anno,
        transform=transform
    )