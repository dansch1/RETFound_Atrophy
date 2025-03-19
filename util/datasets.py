# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------
import json
import os

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


def build_interval_dataset(is_train, args):
    transform = build_transform(is_train, args)
    dataset = IntervalDataset(args, transform=transform)

    return dataset


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


class IntervalDataset(Dataset):
    def __init__(self, args, transform=None):
        with open(args.annotations, "r") as f:
            self.data = json.load(f)

        self.transform = transform
        self.max_intervals = args.max_intervals

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]["image"]
        image = Image.open(image_path).convert("RGB")
        targets = self.data[idx]["targets"]

        while len(targets) < self.max_intervals:
            targets.append([-100, -100, -100])
        targets = torch.tensor(targets[:self.max_intervals + 1], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, targets
