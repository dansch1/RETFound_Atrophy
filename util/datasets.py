# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------
import json
import os
import pathlib

import numpy as np
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

    return dataset(is_train=is_train == 'train', args=args, root=root, transform=transform)


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
    def __init__(self, is_train, args, root, transform=None, aug_config=None):
        self.is_train = is_train
        self.max_intervals = args.max_intervals
        self.transform = transform

        self.data = [path for path in pathlib.Path(root).rglob("*.*")]

        with open(args.annotations) as f:
            self.annotations = json.loads(f.read())

        if aug_config is None:
            aug_config = dict(
                flip_prob=0.5,
                noise_prob=0.3
            )

        self.aug_config = aug_config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert("RGB")

        # make sure len(targets) >= self.max_intervals
        targets = self.annotations.get(image_path.name, [])[:self.max_intervals]

        # random horizontal flip
        if torch.rand(1).item() < self.aug_config["flip_prob"]:
            image = transforms.functional.hflip(image)
            width = image.width
            targets = [[width - x1, width - x0, c] for [x0, x1, c] in targets]  # flip x0, x1

        # random gaussian noise
        if torch.rand(1).item() < self.aug_config["noise_prob"]:
            image = self.add_noise(image)

        if self.transform:
            image = self.transform(image)

        while len(targets) < self.max_intervals:
            targets.append([0, 0, 0])

        targets = torch.tensor(targets, dtype=torch.float32)

        return image, targets

    def add_noise(self, image):
        np_img = np.array(image).astype(np.float32)
        noise = np.random.normal(loc=0, scale=5, size=np_img.shape)
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(np_img)


class IDataset(ICDataset):
    def __getitem__(self, idx):
        image, targets = super().__getitem__(idx)
        return image, targets[:, :-1]
