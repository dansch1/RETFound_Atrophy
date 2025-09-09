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
            x = x[:, 1:, :].mean(dim=1, keepdim=True)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward_features_tokens(self, x):
        """Return per-patch tokens (without CLS), shape [B, N, D]."""
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+N, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) if not self.global_pool else x  # falls du global_pool nutzt, lass es weg
        return x[:, 1:, :]  # [B, N, D]


class IDetector(VisionTransformer):
    def __init__(self, max_intervals, **kwargs):
        super().__init__(**kwargs)

        self.img_size = kwargs['img_size']
        self.max_intervals = max_intervals

        self.head = nn.Linear(self.embed_dim, 2 * max_intervals)  # (x0, x1) for each interval

    def forward(self, x):
        x = self.forward_features(x)

        interval_preds = self.head(x).squeeze(1)  # shape: (batch_size, max_intervals * 2)

        interval_preds = interval_preds.view(-1, self.max_intervals, 2)
        interval_preds = torch.sigmoid(interval_preds) * self.img_size

        return interval_preds


class ICDetector(VisionTransformer):
    def __init__(self, max_intervals, decoder_nhead=8, decoder_num_layers=1, **kwargs):
        super().__init__(**kwargs)
        self.img_size = kwargs['img_size']
        self.max_intervals = max_intervals
        self.embed_dim = kwargs['embed_dim']

        # Learnable queries, one per interval
        self.query_embed = nn.Embedding(max_intervals, self.embed_dim)

        # Small decoder for cross-attention from queries -> patch tokens
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=decoder_nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_num_layers)

        # Heads per query
        self.head_interval = nn.Linear(self.embed_dim, 2)  # (x0, x1), sigmoid -> [0,1], scale -> img size
        self.head_class = nn.Linear(self.embed_dim, self.num_classes)  # class logits (no softmax here)

    def forward(self, x):
        # 1) Per-patch Tokens (keys/values for cross-attention)
        tokens = self.forward_features_tokens(x)  # [B, N, D]
        B, N, D = tokens.shape

        # 2) Queries as seeds for each slot
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, Q, D]

        # 3) Cross-attention: queries attend to tokens
        out = self.decoder(tgt=queries, memory=tokens)  # [B, Q, D]

        # 4) Predict intervals and class logits per query
        interval_raw = torch.sigmoid(self.head_interval(out)) * self.img_size  # [B, Q, 2] in pixels
        # enforce x0 <= x1
        x0 = torch.min(interval_raw[..., 0], interval_raw[..., 1]).unsqueeze(-1)
        x1 = torch.max(interval_raw[..., 0], interval_raw[..., 1]).unsqueeze(-1)
        interval_preds = torch.cat([x0, x1], dim=-1)  # [B, Q, 2]

        class_logits = self.head_class(out)  # [B, Q, num_classes]

        return interval_preds, class_logits


def RETFound_mae(args, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def RETFound_dinov2(args, **kwargs):
    model = timm.create_model(
        'vit_large_patch14_dinov2.lvd142m',
        pretrained=True,
        img_size=224,
        **kwargs
    )
    return model


def I_detector(args, **kwargs):
    model = IDetector(max_intervals=args.max_intervals, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                      mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def IC_detector(args, **kwargs):
    model = ICDetector(max_intervals=args.max_intervals, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                       mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
