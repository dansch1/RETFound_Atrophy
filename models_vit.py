# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


class IDetector(VisionTransformer):
    def __init__(self, max_intervals, **kwargs):
        super().__init__(**kwargs)

        self.max_intervals = max_intervals
        self.head = nn.Linear(self.embed_dim, 2 * max_intervals)  # (x0, x1) for each interval

    def forward(self, x):
        x = self.forward_features(x)

        interval_preds = self.head(x)
        interval_preds = interval_preds.view(-1, self.max_intervals, 2)

        return interval_preds


class ICDetector(VisionTransformer):
    def __init__(self, max_intervals, **kwargs):
        super().__init__(**kwargs)

        self.max_intervals = max_intervals
        # (x0, x1) + num_classes for each interval
        self.head = nn.Linear(self.embed_dim, (2 + self.num_classes) * self.max_intervals)

    def forward(self, x):
        x = self.forward_features(x)

        preds = self.head(x)  # Shape: (batch_size, max_intervals * (2 + num_classes))

        interval_preds = preds[:, :self.max_intervals * 2].view(-1, self.max_intervals, 2)
        class_preds = preds[:, self.max_intervals * 2:].view(-1, self.max_intervals, self.num_classes)

        return interval_preds, class_preds


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def I_detector(**kwargs):
    model = IDetector(max_intervals=10, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def IC_detector(**kwargs):
    model = ICDetector(max_intervals=10, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
