import argparse
import logging
import os
import PIL
import tarfile

import tqdm
import numpy as np
import torch
import torch.utils.data as data
import torchvision

from urllib.request import urlretrieve

import oodd

from oodd.datasets import transforms
from oodd.datasets import BaseDataset
from oodd.constants import TRAIN_SPLIT, VAL_SPLIT, DATA_DIRECTORY, DATA_PATH


LOGGER = logging.getLogger(__file__)

FFHQ_DIRECTORY = f"{DATA_PATH}/ffhq"

class FFHQ32(data.Dataset):
    """Base level FFHQ32 dataset"""

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.root = os.path.join(root, "ffhq-32.npy")
        trX = np.load(self.root, mmap_mode='r')
        np.random.seed(5)
        tr_va_split_indices = np.random.permutation(trX.shape[0])
        if train:
            self.data = trX[tr_va_split_indices[:-7000]]
        else:
            self.data = trX[tr_va_split_indices[-7000:]]

        # self.shuffle(seed=19690720)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (image, target) where target is idx of the target class.
        """
        example = self.data[idx]
        target = 0 # no labels?

        if self.transform is not None:
            example = self.transform(example)

        if self.target_transform is not None:
            target = self.target_transform(0)

        return example, target

    def __repr__(self):
        root, train, transform, target_transform = (
            self.root,
            self.train,
            self.transform,
            self.target_transform
        )
        fmt_str = f"FFHQ32({root=}, {train=}, {transform=}, {target_transform=})"
        return fmt_str

    def __len__(self):
        return len(self.data)


class FFHQ32Quantized(BaseDataset):
    """FFHQ32 dataset serving quantized pixel values in [0, 1] (256 unique values)"""

    _data_source = FFHQ32
    _split_args = {TRAIN_SPLIT: {"train": True}, VAL_SPLIT: {"train": False}}

    default_transform = torchvision.transforms.ToTensor()

    def __init__(
        self,
        split=TRAIN_SPLIT,
        root=FFHQ_DIRECTORY,
        transform=None,
        target_transform=None,
    ):
        super().__init__()

        transform = self.default_transform if transform is None else transform
        self.dataset = self._data_source(
            **self._split_args[split], root=root, transform=transform, target_transform=target_transform
        )

    @classmethod
    def get_argparser(cls):
        parser = argparse.ArgumentParser(description=cls.__name__)
        parser.add_argument("--root", type=str, default=FFHQ_DIRECTORY, help="Data storage location")
        return parser

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class FFHQ32Dequantized(FFHQ32Quantized):
    """FFHQ32 dataset serving dequantized pixel values in [0, 1] via 'x <- (x + u) / (255 + 1))'"""

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            transforms.Scale(a=0, b=255, min_val=0, max_val=1),  # Scale to [0, 1]
            transforms.Dequantize(),  # Add U(0, 1) noise, becomes [0, 256]
            transforms.Scale(a=0, b=1, min_val=0, max_val=256),  # Scale to [0, 1]
        ]
    )
#
#
# class FFHQ32Binarized(FFHQ32Quantized):
#     """FFHQ32 dataset serving binarized pixel values in {0, 1} via """
#
#     default_transform = torchvision.transforms.Compose(
#         [
#             torchvision.transforms.ToTensor(),
#             transforms.Binarize(resample=True),
#         ]
#     )


