# ALQ

Adaptive Loss-aware Quantization for Multi-bit Networks

### Introduction
This repository contains the code of ALQ introduced in our CVPR2020 paper:

Z. Qu, Z. Zhou, Y. Cheng and L. Thiele. Adaptive Loss-aware Quantization for Multi-bit Networks.  

You can also find the corresponding pseudocode in Appendix of the paper.

[PDF](https://arxiv.org/pdf/1912.08883.pdf)

### Dependencies

+ Python 3.7+
+ PyTorch 1.3.1+
+ NVIDIA GPU + CUDA CuDNN (CUDA 10.0)

### Usage

Both MNIST and CIFAR10 datasets can be automatically downloaded via Pytorch.

ILSVRC12 dataset should be downloaded and decompressed into the structure like,

    dir/
      train/
        n01440764/
          n01440764_10026.JPEG
          ...
        ...
      val/
        ILSVRC2012_val_00000001.JPEG
        ...
You may follow some instructions provided in https://pytorch.org/docs/1.1.0/_modules/torchvision/datasets/imagenet.html

To quantize the weights of LeNet5 (on MNIST) by ALQ run

    python lenet5.py --PRETRAIN --ALQ --POSTTRAIN  

To quantize the weights of VGG (on CIFAR10) by ALQ run

    python vgg.py --PRETRAIN --ALQ --POSTTRAIN  

To quantize the weights of ResNet18/34 (on ILSVRC12) by ALQ run

    python resnet.py --PRETRAIN --DOWNLOAD --ALQ --POSTTRAIN --net resnet18 
    
    python resnet.py --PRETRAIN --DOWNLOAD --ALQ --POSTTRAIN --net resnet34
    
    
For more options, please refer to `python xxx.py -h` respectively.

### Results

Model|Weight Bitwidth|Weight Storage|Top-1(%)
:---:|:---:|:---:|:---:
LeNet5|0.40|22.7KB|99.1
VGG|0.66|1.29MB|92.0
ResNet18|2.00|3.44MB|70.0
ResNet34|2.00|6.37MB|73.6


More results can be found in the paper.

### Citation
If you use the code in your research, please cite as

    @InProceedings{Qu_2020_CVPR,
        author = {Qu, Zhongnan and Zhou, Zimu and Cheng, Yun and Thiele, Lothar},
        title = {Adaptive Loss-Aware Quantization for Multi-Bit Networks},
        booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2020}
    }

