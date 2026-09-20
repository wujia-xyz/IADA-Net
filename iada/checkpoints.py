"""Load tensor checkpoints and the retained compact query-adapter format."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from .model import IADANet
from .reader_grades import diagnosis_state


def load_payload(path):
    allowed = [(np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
               np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def state_dict(payload):
    state = payload.get('model_state_dict', payload.get('state_dict', payload))
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Checkpoint does not contain a tensor state dictionary')
    return {k.removeprefix('module.'): v for k, v in state.items()}


def load_model(checkpoint, device='cpu', adapter=None):
    payload = load_payload(checkpoint)
    state = state_dict(payload)
    if any(key.startswith('reader_ordinal.') for key in state):
        state = diagnosis_state(state)
    has_query = 'row_pooling.context_proj.weight' in state
    has_readout = 'depth_pool_score.weight' in state
    variant = 'both' if has_query and has_readout else 'query' if has_query else 'readout' if has_readout else 'fixed'
    if adapter:
        if variant != 'fixed':
            raise ValueError('Compact query adapters require the original fixed-query base checkpoint')
        a = load_payload(adapter)
        if a.get('format') != 'DABI_QUERY_ADAPTER_ONLY' or a.get('format_version') != 1:
            raise ValueError('Unsupported compact adapter format')
        expected = a.get('original_checkpoint', {}).get('sha256')
        digest = hashlib.sha256()
        with Path(checkpoint).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
        actual = digest.hexdigest()
        if expected != actual:
            raise ValueError('Adapter is bound to a different base checkpoint')
        addition = a['adapter_state_dict']
        shapes = {'row_pooling.context_proj.weight': (768, 768), 'row_pooling.context_proj.bias': (768,)}
        if set(addition) != set(shapes) or any(tuple(addition[k].shape) != v or not torch.isfinite(addition[k]).all() for k, v in shapes.items()):
            raise ValueError('Invalid query adapter tensors')
        state.update(addition)
        variant = 'query'
    model = IADANet(variant=variant)
    model.load_state_dict(state, strict=True)
    return model.to(device).float().eval(), payload
