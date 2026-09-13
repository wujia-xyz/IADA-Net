import copy
import numpy as np
import pytest
import torch
from torch import nn
from iada.model import IADANet
from iada.heads import build_head
from iada.train import ranking_loss
from iada.probes import permutation_indices,reindex,magnitude_matched_query,centered_logit_change,image_query,forward_with_query

torch.set_num_threads(2)

class FeatureBackbone(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('features',torch.randn(1,256,768))
    def forward_features(self,x):return {'x_norm_patchtokens':self.features.expand(len(x),-1,-1)}

def test_zero_query_adapter_recovers_base_and_freezes_original():
    torch.manual_seed(7);base=IADANet('fixed',backbone=FeatureBackbone()).eval()
    query=IADANet('query',backbone=FeatureBackbone());query.load_base_state(base.state_dict());query.enable_adaptation()
    x=torch.zeros(2,3,224,224)
    # Equivalent pooling expressions can differ at FP32 rounding precision.
    torch.testing.assert_close(query(x),base(x),rtol=1e-5,atol=1e-6)
    trainable={k for k,p in query.named_parameters() if p.requires_grad}
    assert trainable=={'row_pooling.context_proj.weight','row_pooling.context_proj.bias'}
    assert sum(p.numel() for p in query.parameters() if p.requires_grad)==590592
    assert not query.dinov2.training and not query.interaction_layers[0].training
    loss=torch.nn.functional.cross_entropy(query(x),torch.tensor([0,1]));loss.backward()
    assert all(p.grad is None for k,p in query.named_parameters() if k not in trainable)
    assert query.row_pooling.context_proj.weight.grad.abs().sum()>0

def test_base_state_rejects_missing_tensor():
    model=IADANet('query',backbone=FeatureBackbone());state=IADANet('fixed',backbone=FeatureBackbone()).state_dict()
    del state['classifier.bias']
    with pytest.raises(ValueError):model.load_base_state(state)

def test_depth_head_lateral_invariance_and_capacity():
    torch.manual_seed(9);tokens=torch.randn(2,257,768);idx=permutation_indices('synthetic','1')['lateral_shuffle']
    counts=[]
    for kind in ['gap','gated','cls_mean','depth']:
        model=build_head(kind).eval();counts.append(sum(p.numel() for p in model.parameters()))
        torch.testing.assert_close(model(tokens),model(reindex(tokens,idx)),rtol=1e-5,atol=1e-6)
    assert max(counts)==6339282 and (max(counts)-min(counts))/max(counts)<.004

def test_query_override_and_magnitude_matching():
    torch.manual_seed(11);model=IADANet('query',backbone=FeatureBackbone()).eval();patches=torch.randn(2,256,768)
    reference=image_query(model,patches);own=reference+torch.randn_like(reference)*.1;donor=reference+torch.randn_like(reference)*.2
    torch.testing.assert_close(model.forward_patches(patches),forward_with_query(model,patches,reference))
    matched=magnitude_matched_query(model,patches,own,donor,reference)
    a=centered_logit_change(model,patches,own,reference).square().mean((1,2))
    b=centered_logit_change(model,patches,matched,reference).square().mean((1,2))
    torch.testing.assert_close(a,b)

def test_ranking_loss_single_class_and_order():
    x=torch.tensor([[0.,2.],[0.,-2.]],requires_grad=True)
    assert ranking_loss(x,torch.tensor([1,0]))<ranking_loss(x,torch.tensor([0,1]))
    zero=ranking_loss(x,torch.tensor([1,1]));zero.backward();assert zero.item()==0
