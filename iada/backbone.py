"""Checkpoint-compatible DINOv2 ViT-B/14 with its original positional table."""
import math
import timm
import torch
from torch import nn
from torch.nn import functional as F


class DINOv2Backbone(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        net = timm.create_model('vit_base_patch14_dinov2', pretrained=False,
                                img_size=224, num_classes=0, global_pool='token')
        self.embed_dim, self.patch_size = 768, 14
        self.patch_embed, self.blocks, self.norm = net.patch_embed, net.blocks, net.norm
        self.cls_token = nn.Parameter(net.cls_token.detach().clone())
        self.pos_embed = nn.Parameter(torch.empty(1, 1370, 768))
        self.mask_token = nn.Parameter(torch.zeros(1, 768))
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        if weights is not None:
            state = torch.load(weights, map_location='cpu', weights_only=True)
            if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
                raise ValueError('Expected the official tensor-only DINOv2 ViT-B/14 weights')
            self.load_state_dict(state, strict=True)

    def interpolate_pos_encoding(self, tokens, height, width):
        count = self.pos_embed.shape[1] - 1
        if tokens.shape[1] - 1 == count and height == width:
            return self.pos_embed
        grid = math.isqrt(count)
        rows, columns = height // self.patch_size, width // self.patch_size
        position = self.pos_embed.float()
        patches = position[:, 1:].reshape(1, grid, grid, self.embed_dim)
        patches = F.interpolate(patches.permute(0, 3, 1, 2), mode='bicubic', antialias=False,
                                scale_factor=((rows + .1) / grid, (columns + .1) / grid))
        if patches.shape[-2:] != (rows, columns):
            raise ValueError('Unexpected DINOv2 positional interpolation shape')
        patches = patches.permute(0, 2, 3, 1).reshape(1, rows * columns, self.embed_dim)
        return torch.cat([position[:, :1], patches], dim=1).to(tokens.dtype)

    def forward_features(self, images):
        _, _, height, width = images.shape
        patches = self.patch_embed(images)
        tokens = torch.cat([self.cls_token.expand(images.shape[0], -1, -1), patches], dim=1)
        tokens = tokens + self.interpolate_pos_encoding(tokens, height, width)
        for block in self.blocks:
            tokens = block(tokens)
        normalized = self.norm(tokens)
        return {'x_norm_clstoken': normalized[:, 0], 'x_norm_patchtokens': normalized[:, 1:]}

    def forward(self, images):
        return self.forward_features(images)['x_norm_clstoken']
