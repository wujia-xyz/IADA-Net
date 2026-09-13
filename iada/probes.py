"""Feature-order and image-conditioned-query controls for fixed models."""
import hashlib
import numpy as np
import torch

def permutation_indices(dataset,case_id):
    seed=int.from_bytes(hashlib.sha256(f'eaai-order-probe-v1:{dataset}:{case_id}'.encode()).digest()[:8],'big')
    rng=np.random.default_rng(seed);grid=np.arange(256).reshape(16,16)
    return {'lateral_shuffle':np.stack([row[rng.permutation(16)] for row in grid]).reshape(-1),
            'depth_shuffle':grid[rng.permutation(16)].reshape(-1),
            'depth_reverse':grid[::-1].reshape(-1),'transpose':grid.T.reshape(-1)}

def reindex(tokens,indices):
    """Preserve the CLS token and reorder the 256 patch tokens."""
    idx=torch.as_tensor(np.array(indices,copy=True),device=tokens.device,dtype=torch.long)
    return torch.cat([tokens[:,:1],tokens[:,1:][:,idx]],dim=1)

def image_query(model,patches):
    return model.row_pooling.query[:,0]+model.row_pooling.context_proj(patches.mean(1))

def centered_logit_change(model,patches,query,reference):
    keys=model.row_pooling.key_proj(patches).reshape(-1,16,16,768)
    delta=(keys*(query-reference)[:,None,None,:]).sum(-1)*model.row_pooling.scale
    return delta-delta.mean(-1,keepdim=True)

def magnitude_matched_query(model,patches,own,donor,reference):
    own_rms=centered_logit_change(model,patches,own,reference).square().mean((1,2)).sqrt()
    donor_rms=centered_logit_change(model,patches,donor,reference).square().mean((1,2)).sqrt()
    if torch.any(donor_rms<=1e-12):raise ValueError('Degenerate donor logit change')
    return reference+(donor-reference)*(own_rms/donor_rms)[:,None]

def forward_with_query(model,patches,query):
    """Override only the query residual; leave the target features and classifier fixed."""
    residual=query-model.row_pooling.query[:,0]
    handle=model.row_pooling.context_proj.register_forward_hook(lambda module,args,output:residual.expand_as(output))
    try:return model.forward_patches(patches)
    finally:handle.remove()
