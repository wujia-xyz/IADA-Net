"""Portable full-image preprocessing and dataset manifests."""
from pathlib import Path
import csv,json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

MEAN=(.485,.456,.406)
STD=(.229,.224,.225)

def read_records(path):
    path=Path(path)
    if path.suffix.lower()=='.csv':
        with path.open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
    else:
        records=json.loads(path.read_text(encoding='utf-8'))
        if isinstance(records,dict):records=records['records']
    for r in records:
        r['label']=int(r['label'])
        if r['label'] not in (0,1):raise ValueError('Labels must be 0=benign, 1=malignant')
        if r.get('fold') not in ('',None):r['fold']=int(r['fold'])
    return records

def read_rgb(path):
    with Image.open(path) as image:return np.asarray(image.convert('RGB'))

def transform(stage=None,seed=42):
    ops=[]
    if stage=='base':
        ops=[A.HorizontalFlip(p=.5),A.VerticalFlip(p=.2),
             A.RandomBrightnessContrast(brightness_limit=.2,contrast_limit=.2,p=.5),
             A.GaussNoise(std_range=(.02,.1),p=.3),A.GaussianBlur(blur_limit=(3,5),p=.2),
             A.Rotate(limit=15,p=.5),A.Affine(translate_percent={'x':(-.1,.1),'y':(-.1,.1)},scale=(.9,1.1),p=.3)]
    elif stage=='adapt':
        ops=[A.HorizontalFlip(p=.5),A.RandomBrightnessContrast(brightness_limit=.15,contrast_limit=.15,p=.5),
             A.Rotate(limit=10,border_mode=cv2.BORDER_REFLECT_101,p=.4),
             A.Affine(scale=(.95,1.05),translate_percent=(-.03,.03),border_mode=cv2.BORDER_REFLECT_101,p=.3)]
    elif stage is not None:raise ValueError(stage)
    return A.Compose(ops+[A.Normalize(mean=MEAN,std=STD),ToTensorV2()],seed=seed)

def image_tensor(path):
    rgb=cv2.resize(read_rgb(path),(224,224),interpolation=cv2.INTER_LINEAR)
    return transform()(image=rgb)['image']

class Images(Dataset):
    def __init__(self,records,root,stage=None,seed=42):
        self.records=records;self.root=Path(root);self.transform=transform(stage,seed)
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        r=self.records[index];path=Path(r['image_path']);path=path if path.is_absolute() else self.root/path
        image=cv2.resize(read_rgb(path),(224,224),interpolation=cv2.INTER_LINEAR)
        return self.transform(image=image)['image'],r['label'],index

def worker_seed(worker_id):
    import torch,random
    seed=torch.initial_seed()%2**32
    random.seed(seed);np.random.seed(seed);cv2.setNumThreads(0)
    torch.utils.data.get_worker_info().dataset.transform.set_random_seed(seed)
