import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from scipy.special import expit

from iada import checkpoints, predict_manifest
from iada.model import IADANet
from iada.reader_grades import attach_reader_head


class FeatureBackbone(nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer("features",torch.randn(1,256,768))

    def forward_features(self,images):
        return {"x_norm_patchtokens":self.features.expand(len(images),-1,-1)}


def test_checkpoint_loader_discards_only_the_complete_reader_head(tmp_path,monkeypatch):
    monkeypatch.setattr(checkpoints,"IADANet",lambda variant: IADANet(variant,backbone=FeatureBackbone()))
    model=IADANet("fixed",backbone=FeatureBackbone()).eval();attach_reader_head(model)
    path=tmp_path/"graded.pt";torch.save({"model_state_dict":model.state_dict()},path)
    loaded,_=checkpoints.load_model(path)
    images=torch.zeros(2,3,224,224)
    torch.testing.assert_close(loaded(images),model(images),rtol=0,atol=0)
    assert not hasattr(loaded,"reader_ordinal")
    state=model.state_dict();state.pop("reader_ordinal.raw_gaps")
    incomplete=tmp_path/"incomplete.pt"
    torch.save({"model_state_dict":state},incomplete)
    with pytest.raises(ValueError,match="schema"):
        checkpoints.load_model(incomplete)


def test_bound_legacy_adapter_still_loads(tmp_path,monkeypatch):
    monkeypatch.setattr(checkpoints,"IADANet",lambda variant: IADANet(variant,backbone=FeatureBackbone()))
    model=IADANet("fixed",backbone=FeatureBackbone()).eval()
    base=tmp_path/"base.pt";torch.save({"model_state_dict":model.state_dict()},base)
    adapter=tmp_path/"adapter.pt"
    payload={"format":"DABI_QUERY_ADAPTER_ONLY","format_version":1,
             "original_checkpoint":{"sha256":hashlib.sha256(base.read_bytes()).hexdigest()},
             "adapter_state_dict":{"row_pooling.context_proj.weight":torch.zeros(768,768),"row_pooling.context_proj.bias":torch.zeros(768)}}
    torch.save(payload,adapter)
    loaded,_=checkpoints.load_model(base,adapter=adapter)
    images=torch.zeros(2,3,224,224)
    torch.testing.assert_close(loaded(images),model(images),rtol=1e-5,atol=1e-6)
    payload["original_checkpoint"]["sha256"]="0"*64;torch.save(payload,adapter)
    with pytest.raises(ValueError,match="different base"):
        checkpoints.load_model(base,adapter=adapter)


def test_manifest_predictions_preserve_order_and_fp32_margin(tmp_path,monkeypatch):
    class Model(nn.Module):
        def forward(self,images):
            value=images[:,0,0,0]
            return torch.stack([value,torch.ones_like(value)],dim=1)
    checkpoint=tmp_path/"synthetic.pt";checkpoint.write_bytes(b"synthetic checkpoint fixture")
    manifest=tmp_path/"images.csv"
    manifest.write_text("sample_id,image_path\nsecond,2.bin\ntie,1.bin\nfirst,0.bin\n",encoding="utf-8")
    monkeypatch.setattr(predict_manifest,"load_model",lambda *args:(Model(),{}))
    monkeypatch.setattr(predict_manifest,"image_tensor",lambda path: torch.full((3,2,2),float(Path(path).stem)))
    output=tmp_path/"probabilities.csv"
    predict_manifest.main(["--checkpoint",str(checkpoint),"--manifest",str(manifest),"--output",str(output),"--batch-size","2"])
    with output.open() as stream: actual=list(csv.DictReader(stream))
    assert [row["sample_id"] for row in actual]==["second","tie","first"]
    np.testing.assert_array_equal([float(row["score"]) for row in actual],[-1.,0.,1.])
    np.testing.assert_array_equal([float(row["probability"]) for row in actual],expit(np.asarray([-1.,0.,1.])))
    receipt=json.loads(output.with_suffix(".csv.json").read_text())
    assert receipt["images"]==3 and receipt["diagnosis_labels_used"] is False
    with pytest.raises(SystemExit):
        predict_manifest.main(["--checkpoint",str(checkpoint),"--manifest",str(manifest),"--output",str(output)])
