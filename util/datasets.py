# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------
import json
import os
import pathlib

import torch
from PIL import Image
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset


def build_I_dataset(is_train, args):
    return _build_custom_dataset(is_train, args, IDataset)


def build_IC_dataset(is_train, args):
    return _build_custom_dataset(is_train, args, ICDataset)


def _build_custom_dataset(is_train, args, dataset):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)

    return dataset(root, args, transform=transform)


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train == 'train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class ICDataset(Dataset):
    def __init__(self, root, args, transform=None):
        self.data = [path for path in pathlib.Path(root).rglob("*.*")]

        with open(args.annotations) as f:
            self.annotations = json.loads(f.read())

        self.transform = transform
        self.max_intervals = args.max_intervals

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        targets = self.annotations.get(image_path.name, [])

        while len(targets) < self.max_intervals:
            targets.append([-1, -1, 0])

        targets = torch.tensor(targets[:self.max_intervals + 1], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, targets


class IDataset(ICDataset):
    def __getitem__(self, idx):
        image, targets = super().__getitem__(idx)
        return image, targets[:, :-1]
