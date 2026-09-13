"""IADA-Net and its fixed-query/readout adaptation controls."""
import torch
from torch import nn
from .backbone import DINOv2Backbone
from .layers import AttentionRowPooling, ConditionalRowPooling, SimpleDepthEncoding, BidirectionalInteractionLayer

VARIANTS = ('fixed', 'query', 'readout', 'both')


class IADANet(nn.Module):
    def __init__(self, variant='query', backbone_weights=None, backbone=None):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant: {variant}')
        self.variant = variant
        self.adaptation = False
        self.dinov2 = backbone if backbone is not None else DINOv2Backbone(backbone_weights)
        self.row_pooling = (ConditionalRowPooling if variant in ('query', 'both') else AttentionRowPooling)(768, 4)
        self.depth_encoding = SimpleDepthEncoding(64, 16)
        self.input_proj = nn.Sequential(nn.Linear(832, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(.3))
        self.interaction_layers = nn.ModuleList([BidirectionalInteractionLayer(256, 4, 16, .3) for _ in range(2)])
        self.fusion = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU())
        self.classifier = nn.Linear(256, 2)
        if variant in ('readout', 'both'):
            self.depth_pool_score = nn.Linear(256, 1)
            nn.init.zeros_(self.depth_pool_score.weight)
            nn.init.zeros_(self.depth_pool_score.bias)

    def load_base_state(self, state):
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        allowed = set()
        if self.variant in ('query', 'both'):
            allowed |= {'row_pooling.context_proj.weight', 'row_pooling.context_proj.bias'}
        if self.variant in ('readout', 'both'):
            allowed |= {'depth_pool_score.weight', 'depth_pool_score.bias'}
        result = self.load_state_dict(state, strict=False)
        if set(result.missing_keys) != allowed or result.unexpected_keys:
            raise ValueError(f'Base-checkpoint schema mismatch: {result}')

    def enable_adaptation(self):
        if self.variant == 'fixed':
            raise ValueError('Choose query, readout, or both for the adaptation stage')
        self.requires_grad_(False)
        if self.variant in ('query', 'both'):
            self.row_pooling.context_proj.requires_grad_(True)
        if self.variant in ('readout', 'both'):
            self.depth_pool_score.requires_grad_(True)
        self.adaptation = True
        self.train()
        return self

    def train(self, mode=True):
        if not self.adaptation:
            return super().train(mode)
        super().train(False)
        if self.variant in ('query', 'both'):
            self.row_pooling.context_proj.train(mode)
        if self.variant in ('readout', 'both'):
            self.depth_pool_score.train(mode)
        return self

    def forward_patches(self, patches):
        batch = len(patches)
        if patches.shape[1:] != (256, 768):
            raise ValueError('Expected 256 patch features with 768 channels')
        rows = self.row_pooling(patches.reshape(batch, 16, 16, 768))
        codes = self.depth_encoding(torch.arange(16, device=patches.device))[None].expand(batch, -1, -1)
        sequence = self.input_proj(torch.cat([rows, codes], dim=-1))
        td, bu = sequence, sequence.flip(1)
        for layer in self.interaction_layers:
            td, bu = layer(td, bu)
        fused = self.fusion(torch.cat([td, bu.flip(1)], dim=-1))
        pooled = fused.mean(1)
        if self.variant in ('readout', 'both'):
            weights = self.depth_pool_score(fused).softmax(1)
            pooled = pooled + ((weights - 1 / 16) * fused).sum(1)
        return self.classifier(pooled)

    def forward(self, images):
        if images.shape[1:] != (3, 224, 224):
            raise ValueError('Expected normalized RGB inputs with shape B x 3 x 224 x 224')
        if self.adaptation:
            with torch.no_grad():
                patches = self.dinov2.forward_features(images)['x_norm_patchtokens']
        else:
            patches = self.dinov2.forward_features(images)['x_norm_patchtokens']
        return self.forward_patches(patches)
