# DSSLIC
Pytorch implmementation of DSSLIC: Deep Semantic Segmentation-based Layered Image Compression

Paper: https://arxiv.org/abs/1806.03348

## Original Requirements: 
- Ubuntu 16.04
- Python 2.7
- Cuda 8.0
- Pyorch 0.3.0

## My Environments:
- GTX 1080 Ti (with 11GB graphic memory)
- Ubuntu 16.04
- Python 3.5
- Cuda 9.0
- Pytorch 0.4.1

# Training
## ADE20K Dataset
python3 train.py --name ADE20K_model --dataroot ./datasets/ADE20K/ --label_nc 151 --loadSize 256 --resize_or_crop resize --batchSize 8
## Cityscapes Dataset
python3 train.py --name Cityscapes_model --dataroot ./datasets/cityscapes/ --label_nc 35 --loadSize 512 --resize_or_crop scale_width --batchSize 1

# Testing
## ADE20K testset
python3 test.py --name ADE20K_model --dataroot ./datasets/ADE20K/ --label_nc 151 --resize_or_crop none --batchSize 1 --how_many 50
## Kodak testset
python3 test.py --name ADE20K_model --dataroot ./datasets/Kodak/ --label_nc 151 --resize_or_crop none --batchSize 1 --how_many 24
## Cityscapes testset
python3 test.py --name Cityscapes_model --dataroot ./datasets/cityscapes/ --label_nc 35 --loadSize 1024 --resize_or_crop scale_width --batchSize 1 --how_many 50
