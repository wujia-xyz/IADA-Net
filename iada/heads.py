"""Capacity-matched heads from the retained controlled study."""
import torch
from torch import nn
from . import layers as original

class DepthHead(nn.Module):
    def __init__(self):
        super().__init__();self.row_pooling=original.AttentionRowPooling(768,4)
        self.depth_encoding=original.SimpleDepthEncoding(64,16)
        self.input_proj=nn.Sequential(nn.Linear(832,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.3))
        self.interaction_layers=nn.ModuleList([original.BidirectionalInteractionLayer(256,4,16,.3) for _ in range(2)])
        self.fusion=nn.Sequential(nn.Linear(512,256),nn.LayerNorm(256),nn.GELU())
        self.classifier=nn.Linear(256,2)

    def forward(self,tokens):
        patches=tokens[:,1:];b=len(patches);rows=self.row_pooling(patches.reshape(b,16,16,768))
        code=self.depth_encoding(torch.arange(16,device=patches.device))[None].expand(b,-1,-1)
        h=self.input_proj(torch.cat([rows,code],dim=-1));td,bu=h,h.flip(1)
        for layer in self.interaction_layers:td,bu=layer(td,bu)
        return self.classifier(self.fusion(torch.cat([td,bu.flip(1)],dim=-1)).mean(1))


class GenericHead(nn.Module):
    def __init__(self,kind,width):
        super().__init__();self.kind=kind;self.width=width
        d=1536 if kind=='cls_mean' else 768
        if kind=='gated':
            self.attn_v=nn.Linear(768,128);self.attn_u=nn.Linear(768,128);self.attn_w=nn.Linear(128,1)
        self.classifier=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,width),nn.GELU(),nn.Dropout(.3),
            nn.Linear(width,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.3),nn.Linear(256,2))

    def forward(self,tokens):
        f=tokens[:,1:]
        if self.kind=='gated':
            a=self.attn_w(torch.tanh(self.attn_v(f))*torch.sigmoid(self.attn_u(f))).softmax(1)
            pooled=(a*f).sum(1)
        else:pooled=f.mean(1)
        if self.kind=='cls_mean':pooled=torch.cat([tokens[:,0],pooled],dim=-1)
        return self.classifier(pooled)


TARGET_PARAMETERS=6339282
KINDS=('gap','gated','cls_mean','depth')


def build_head(kind):
    if kind=='depth':return DepthHead()
    if kind not in KINDS:raise ValueError(kind)
    # Width is determined by capacity alone, before data or results are read.
    d=1536 if kind=='cls_mean' else 768
    scorer=196993 if kind=='gated' else 0
    fixed=2*d+256+512+514+scorer
    width=max(32,round((TARGET_PARAMETERS-fixed)/(d+257)/32)*32)
    head=GenericHead(kind,width)
    assert abs(sum(p.numel() for p in head.parameters())/TARGET_PARAMETERS-1)<.01
    return head
