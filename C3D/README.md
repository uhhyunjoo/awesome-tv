# C3D
- `Original Paper` : [Learning Spatiotemporal Features with 3D Convolutional Networks](https://openaccess.thecvf.com/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf)
- `Referenced repo` : [pytorch-video-recognition](https://github.com/jfzhang95/pytorch-video-recognition)
- `What did I do?` : Train C3D from scratch with UCF101 (the paper trained C3D with Sports-1M)

## What is this paper about?

This paper proposed a simple, yet effective approach for spatio-temporal feature learning using deep 3-dimensional convolutional networks (3D ConvNets) trained on a large scale supervised video dataset (Sports-1M).

1) 3D ConvNets are more suitable for spatiotemporal feature learning compared to 2D ConvNets.
2) Best setting : A homogeneous architecture with 3x3x3 convolution kernels in all layers.
3) Using C3D features with a linear classifier can outperform or approach sota on different video analysis benchmarks.

## Experiment
- Dataset : UCF101
- Train : UCF101 train split 1 (trainlist01.txt)
- Valiation : UCF101 test split 1 (testlist01.txt)

```
python main.py --learning_rate 0.005 --scheduler step --step_size 20
```

![image](https://user-images.githubusercontent.com/41139770/161973218-47d96b2e-a6c2-4ba7-bbd8-fde3bcffe459.png)

|Epoch|train_acc|train_loss|val_acc|val_loss|
|:----:|:----:|:----:|:----:|:----:|
|60|98.469|0.045|42.162|4.651|



## Environment
```
Collecting environment information...
PyTorch version: 1.6.0+cu101
Is debug build: False
CUDA used to build PyTorch: 10.1
ROCM used to build PyTorch: N/A

OS: Ubuntu 18.04.5 LTS (x86_64)
GCC version: (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
Clang version: Could not collect
CMake version: version 3.10.2
Libc version: glibc-2.17

Python version: 3.7.11 (default, Jul 27 2021, 14:32:16)  [GCC 7.5.0] (64-bit runtime)
Python platform: Linux-4.15.0-39-generic-x86_64-with-debian-buster-sid
Is CUDA available: True
CUDA runtime version: 10.1.243
GPU models and configuration: 
GPU 0: Tesla V100-SXM2-32GB
GPU 1: Tesla V100-SXM2-32GB
GPU 2: Tesla V100-SXM2-32GB
GPU 3: Tesla V100-SXM2-32GB
GPU 4: Tesla V100-SXM2-32GB
GPU 5: Tesla V100-SXM2-32GB
GPU 6: Tesla V100-SXM2-32GB
GPU 7: Tesla V100-SXM2-32GB

Nvidia driver version: 410.104
cuDNN version: /usr/lib/x86_64-linux-gnu/libcudnn.so.7.6.5
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: True

Versions of relevant libraries:
[pip3] numpy==1.18.2
[pip3] pytorchvideo==0.1.5
[pip3] torch==1.6.0+cu101
[pip3] torchinfo==1.6.3
[pip3] torchsummary==1.5.1
[pip3] torchsummaryX==1.3.0
[pip3] torchvision==0.7.0+cu101
[conda] numpy                     1.18.2                   pypi_0    pypi
[conda] torch                     1.6.0+cu101              pypi_0    pypi
[conda] torchvision               0.7.0+cu101              pypi_0    pypi
```
