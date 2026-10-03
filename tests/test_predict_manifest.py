import csv
import torch
from PIL import Image

from iada import predict_manifest


def test_prediction_writes_only_scores_without_hashing_images(tmp_path, monkeypatch):
    image = tmp_path / 'image.png'
    Image.new('RGB', (4, 4), color='gray').save(image)
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'image_path', 'image_sha256'])
        writer.writeheader()
        writer.writerow({'sample_id': 'synthetic', 'image_path': 'image.png', 'image_sha256': 'historical-unused'})

    class Model(torch.nn.Module):
        def forward(self, images):
            return torch.tensor([[0., 1.]]).expand(len(images), -1)

    monkeypatch.setattr(predict_manifest, 'load_model', lambda *args: (Model(), {}))
    from iada import source_protocol
    monkeypatch.setattr(source_protocol, 'file_hash', lambda *args: (_ for _ in ()).throw(AssertionError('Unexpected hash')))
    output = tmp_path / 'scores.csv'
    predict_manifest.main(['--checkpoint', str(tmp_path / 'model.pt'), '--manifest', str(manifest),
                           '--data-root', str(tmp_path), '--output', str(output), '--device', 'cpu'])
    with output.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['sample_id'] == 'synthetic'
    assert float(rows[0]['score']) == 1.
    assert .73 < float(rows[0]['probability']) < .74
    assert not output.with_suffix('.csv.json').exists()
