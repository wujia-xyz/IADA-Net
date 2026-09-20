"""Native bicubic positional forward with a deterministic spatial transpose.

The first derivative uses the actual interpolation basis on the input device.
This is the positional implementation used by the fixed-source study, not a
change to the historical backbone default. Higher-order gradients are excluded.
"""
import math
from types import MethodType
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

_CACHE = {}


def spatial_map(x, scale):
    height, width = x.shape[-2:]
    key = (str(x.device), str(x.dtype), height, width, scale)
    if key not in _CACHE:
        basis = torch.eye(height * width, dtype=x.dtype, device=x.device).reshape(height * width, 1, height, width)
        output = F.interpolate(basis, scale_factor=scale, mode="bicubic", align_corners=False, antialias=False)
        _CACHE[key] = output.reshape(height * width, -1).double().contiguous()
    return _CACHE[key]


class BicubicPosition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sy, sx):
        ctx.shape, ctx.dtype = tuple(x.shape), x.dtype
        ctx.save_for_backward(spatial_map(x, (sy, sx)))
        return F.interpolate(x, scale_factor=(sy, sx), mode="bicubic", align_corners=False, antialias=False)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        matrix, = ctx.saved_tensors
        value = gradient.double().reshape(-1, matrix.shape[1]) @ matrix.T
        return value.reshape(ctx.shape).to(dtype=ctx.dtype), None, None


def positional(self, tokens, height, width):
    count = self.pos_embed.shape[1] - 1
    if tokens.shape[1] - 1 == count and height == width:
        return self.pos_embed
    grid = math.isqrt(count)
    rows, columns = height // self.patch_size, width // self.patch_size
    position = self.pos_embed.float()
    patches = position[:, 1:].reshape(1, grid, grid, self.embed_dim).permute(0, 3, 1, 2)
    patches = BicubicPosition.apply(patches, (rows + .1) / grid, (columns + .1) / grid)
    if patches.shape[-2:] != (rows, columns):
        raise RuntimeError("Unexpected positional interpolation shape")
    patches = patches.permute(0, 2, 3, 1).reshape(1, rows * columns, self.embed_dim)
    return torch.cat([position[:, :1], patches], dim=1).to(tokens.dtype)


def install(backbone):
    backbone.interpolate_pos_encoding = MethodType(positional, backbone)
