# ALQ

Adaptive Loss-aware Quantization for Multi-bit Networks

### Introduction
This repository contains the code of ALQ introduced in our CVPR2020 paper:

Z. Qu, Z. Zhou, Y. Cheng and L. Thiele. Adaptive Loss-aware Quantization for Multi-bit Networks.  

[PDF](https://arxiv.org/pdf/1912.08883.pdf)

### Dependencies

+ Python 3.7+
+ PyTorch 1.3.1+
+ NVIDIA GPU + CUDA CuDNN (CUDA 10.0)

### Usage

Download the ImageNet dataset and decompress into the structure like

    dir/
      train/
        n01440764/
          n01440764_10026.JPEG
          ...
        ...
      val/
        ILSVRC2012_val_00000001.JPEG
        ...

To train a quantized "pre-activation" ResNet-18, simply run

    python imagenet.py --gpu 0,1,2,3 --data /PATH/TO/IMAGENET --mode preact --depth 18 --qw 1 --qa 2 --logdir_id w1a2 

After the training, the result model will be stored in `./train_log/w1a2`.

For more options, please refer to `python imagenet.py -h`. 

### Results

More results can be found in the paper.

### Citation
If you use the code in your research, please cite as

    @inproceedings{bib:CVPR20:Qu,
        author = {Qu, Zhongnan and Zhou, Zimu and Cheng, Yun and Thiele, Lothar},
        title = {Adaptive Loss-aware Quantization for Multi-bit Networks},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2020},
    }
