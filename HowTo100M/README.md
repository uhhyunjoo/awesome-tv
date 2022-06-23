# HowTo100M

- `Original Paper` : [HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips](https://openaccess.thecvf.com/content_ICCV_2019/papers/Miech_HowTo100M_Learning_a_Text-Video_Embedding_by_Watching_Hundred_Million_Narrated_ICCV_2019_paper.pdf)
- `Referenced repo` : [howto100m](https://github.com/antoine77340/howto100m)
- `What did I do?` : Train embedding from scratch on MSR-VTT-1kA (the paper trained embedding on HowTo100M, which is a large scale dataset)

## What is this paper about?

1) introducing a large-scale dataset HowTo100M (contains 136 million video clips sourced from 1.22M narrarted instruction web videos)
2) a text-video embedding trained on HowTo100M leads to sota results for text-to-video retrieval and action localization
3) the embedding transfers well to other domains (fine-tuning on MSR-VTT and LSMDC is better than models trained on these datasets alone)

## Experiment
- Dataset : MSR-VTT-1kA (train, test)
- Task : Clip retrieval
- R@K : Recall (K = 1, 5, 10)
- MR : Median Rank

```
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='7' python train.py --num_thread_reader=1 --epochs=100 --batch_size=32 \
--n_pair=1 --n_display=100 --embd_dim=4096 --checkpoint_dir=ckpt --msrvtt 1 --eval_msrvtt 1 --negative_weighting 0
```

![image](https://user-images.githubusercontent.com/41139770/162184800-84601743-d8e7-4c25-910e-89df6d957658.png)

|Epoch|R@1|R@5|R@10|Median R|
|:----:|:----:|:----:|:----:|:----:|
|83|11.5|34.8|48|11|

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
