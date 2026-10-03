# Third-party components

- [URFM](https://github.com/sonovision-ai/URFM): the final ultrasound-pretrained ViT-L/16 encoder. Obtain its pretraining checkpoint from the authors' [weight repository](https://huggingface.co/QingboKang/URFM), follow its access/license terms, and cite the URFM paper. Weights are not redistributed here.
- [DINOv2](https://github.com/facebookresearch/dinov2): Meta's pretrained visual encoder; original code and weights use the Apache License 2.0. Obtain the weights from the official source and follow that license.
- [timm](https://github.com/huggingface/pytorch-image-models): installed dependency used to construct the ViT blocks, under Apache License 2.0.
- PyTorch, torchvision, Albumentations, OpenCV, NumPy, Pillow and scikit-learn are installed dependencies with their own licenses.
- Dataset images are not redistributed. Consult each dataset's original publication and access terms.

The original DABI-Net operators in this release are the authors' code. Other comparison methods are referenced through their original/predecessor implementations rather than copied into this repository.
