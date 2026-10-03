"""Small CPU checks for the R9 readout and checkpoint dispatch."""
import torch
from torch import nn

from iada import checkpoints
from iada.urfm import URFMIADA


class PatchEncoder(nn.Module):
    embed_dim = 1024

    def forward_features(self, images):
        return {'x_norm_patchtokens': images.new_zeros(len(images), 196, 1024)}


def test_base_to_query_preserves_logits_and_adapter_boundary():
    torch.manual_seed(7)
    base = URFMIADA(PatchEncoder(), 'fixed').eval()
    query = URFMIADA(PatchEncoder(), 'query')
    query.load_base_state(base.state_dict())
    query.enable_adaptation()
    patches = torch.randn(2, 196, 1024)
    torch.testing.assert_close(query.forward_patches(patches), base.forward_patches(patches),
                               rtol=1e-5, atol=1e-6)
    trainable = {name for name, parameter in query.named_parameters() if parameter.requires_grad}
    assert trainable == {'row_pooling.context_proj.weight', 'row_pooling.context_proj.bias'}
    assert sum(parameter.numel() for parameter in query.parameters() if parameter.requires_grad) == 1049600


def test_urfm_full_checkpoint_dispatch_without_legacy_adapter(monkeypatch):
    class TinyModel(nn.Module):
        def __init__(self, variant):
            super().__init__()
            self.variant = variant
            self.encoder = nn.Module()
            self.encoder.net = nn.Linear(1, 1)
            self.row_pooling = nn.Module()
            self.row_pooling.context_proj = nn.Linear(1, 1)

    monkeypatch.setattr('iada.urfm.URFMIADA', TinyModel)
    source = TinyModel('query')
    monkeypatch.setattr(checkpoints, 'load_payload', lambda _: {'model_state_dict': source.state_dict()})
    model, _ = checkpoints.load_model('unused.pt')
    assert isinstance(model, TinyModel) and model.variant == 'query'
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], tensor)
