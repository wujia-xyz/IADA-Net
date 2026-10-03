"""URFM-L/16 encoder and the R9 IADA readout.

The module names match the R9 full-model checkpoints: ``encoder.net`` is the
ViT-L/16 and ``row_pooling.context_proj`` is the query adapter.
"""
import timm
import torch
from torch import nn

from .layers import (AttentionRowPooling, BidirectionalInteractionLayer,
                     ConditionalRowPooling, SimpleDepthEncoding)


class URFMLarge(nn.Module):
    """URFM ViT-L/16 at 224 pixels, exposing its 14 by 14 patch tokens."""

    embed_dim = 1024
    patch_size = 16

    def __init__(self, weights=None, weights_key='ema_state_dict'):
        super().__init__()
        self.net = timm.create_model('vit_large_patch16_224', pretrained=False,
                                     num_classes=0, global_pool='token')
        if weights is not None:
            payload = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
            if weights_key not in payload or not isinstance(payload[weights_key], dict):
                raise ValueError(f'URFM pretraining checkpoint lacks {weights_key}')
            state = {k.removeprefix('module.'): v for k, v in payload[weights_key].items()
                     if not k.removeprefix('module.').startswith(('decoder', 'mask_token'))}
            self.net.load_state_dict(state, strict=True)

    def forward_features(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError('Expected normalized B x 3 x 224 x 224 input')
        tokens = self.net.forward_features(images)
        if tokens.shape[1:] != (197, self.embed_dim):
            raise ValueError('Unexpected URFM token shape')
        return {'x_norm_clstoken': tokens[:, 0], 'x_norm_patchtokens': tokens[:, 1:]}


class URFMIADA(nn.Module):
    """R9 u16_iada: 14-row IADA with explicit depth and a query adapter."""

    def __init__(self, encoder=None, variant='query'):
        super().__init__()
        if variant not in ('fixed', 'query'):
            raise ValueError(f'Unknown URFM IADA variant: {variant}')
        self.variant = variant
        self.adaptation = False
        self.encoder = encoder if encoder is not None else URFMLarge()
        dim, grid = self.encoder.embed_dim, 14
        self.row_pooling = (ConditionalRowPooling if variant == 'query' else AttentionRowPooling)(dim, 4)
        self.depth_encoding = SimpleDepthEncoding(64, grid)
        self.input_proj = nn.Sequential(nn.Linear(dim + 64, 256), nn.LayerNorm(256),
                                        nn.GELU(), nn.Dropout(.3))
        self.interaction_layers = nn.ModuleList(
            [BidirectionalInteractionLayer(256, 4, grid, .3) for _ in range(2)])
        self.fusion = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU())
        self.classifier = nn.Linear(256, 2)

    def load_base_state(self, state):
        allowed = ({'row_pooling.context_proj.weight', 'row_pooling.context_proj.bias'}
                   if self.variant == 'query' else set())
        result = self.load_state_dict(state, strict=False)
        if set(result.missing_keys) != allowed or result.unexpected_keys:
            raise ValueError(f'Base-checkpoint schema mismatch: {result}')

    def enable_adaptation(self):
        if self.variant != 'query':
            raise ValueError('Adaptation requires the query variant')
        self.requires_grad_(False)
        self.row_pooling.context_proj.requires_grad_(True)
        self.adaptation = True
        return self.train()

    def train(self, mode=True):
        if not self.adaptation:
            return super().train(mode)
        super().train(False)
        self.row_pooling.context_proj.train(mode)
        return self

    def forward_patches(self, patches):
        if patches.ndim != 3 or patches.shape[1:] != (196, 1024):
            raise ValueError('Expected B x 196 x 1024 URFM patch features')
        batch = len(patches)
        rows = self.row_pooling(patches.reshape(batch, 14, 14, 1024))
        codes = self.depth_encoding(torch.arange(14, device=patches.device))[None].expand(batch, -1, -1)
        sequence = self.input_proj(torch.cat([rows, codes], dim=-1))
        td, bu = sequence, sequence.flip(1)
        for layer in self.interaction_layers:
            td, bu = layer(td, bu)
        fused = self.fusion(torch.cat([td, bu.flip(1)], dim=-1))
        return self.classifier(fused.mean(1))

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError('Expected normalized B x 3 x 224 x 224 input')
        if self.adaptation:
            with torch.no_grad():
                patches = self.encoder.forward_features(images)['x_norm_patchtokens']
        else:
            patches = self.encoder.forward_features(images)['x_norm_patchtokens']
        return self.forward_patches(patches)
