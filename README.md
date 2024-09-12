# new_SMR

## 运行环境

### 硬设备

推荐使用带有显存大于4G的NVIDIA显卡

推荐内存16G以上



### 支持软件

  Windows、Linux运行环境

### 数据结构

  基于python以及vgg19搭建的各种深度学习神经网络

## 使用过程

### 安装与初始化

#### 需求

​    Linux或windows皆可

Python >= 3.6

CUDA >= 10.0.130 (需要安装nvcc)

  conda环境下安装nvcc: conda install -c nvidia cuda-nvcc

#### 环境安装流程

  创建环境

```
    $ conda create --name smr python=3.7
    $ conda activate smr
```

  安装Pytorch(或更高版本)

```
$ conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```

  安装Kaolin

```
    $ git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
```

​    cd C:\Users\Administrator\kaolin-master

/*进入kaolin文件，路径每个人不一样，只要是kaolin文件中setup.py在的目录就行。*/

python setup.py install

/*设置kaolin*/

  其他需要安装的包: tqdm, trimesh, imageio, tensorboard

####   代码下载

​    [wdnmdzzzad/new_SMR (github.com)](https://github.com/wdnmdzzzad/new_SMR)，此链接下载源码

### 输入

#### 输入数据集

软件根目录下train文件，按照目标物体进行图片分类

例子：

/train/bed

​    -bed1_00.jpg /*bed1的材质*/

​    -bed1_01_1.jpg /*bed1的1频率图第一张*/

​    -bed1_01_2.jpg /*bed1的1频率图第二张*/

​    -bed1_01_3.jpg 

​    -bed1_64_1.jpg /*bed1的64频率图第一张*/

​    -bed1_64_2.jpg /*bed1的64频率图第二张*/

​    -bed1_64_3.jpg

​    ……

​    -bed1_mask.png /*bed1的mask*/

#### 运行方法

以管理员模式运行cmd，进入代码根目录下，运行 python train.py

## 原模型链接

[dvlab-research/SMR: Self-Supervised 3D Mesh Reconstruction from Single Images (CVPR2021) (github.com)](https://github.com/dvlab-research/SMR)

为 CC BY-NC-SA 4.0 公共版权许可，License在项目文件中

## 原作者介绍

```
@InProceedings{Hu_2021_CVPR,
    author    = {Hu, Tao and Wang, Liwei and Xu, Xiaogang and Liu, Shu and Jia, Jiaya},
    title     = {Self-Supervised 3D Mesh Reconstruction From Single Images},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {6002-6011}
}
```

