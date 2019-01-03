# segmentlink_yh
use seglink to detect text
# SegLink

Detecting Oriented Text in Natural Images by Linking Segments (https://arxiv.org/abs/1703.06520).

## Prerequisites

The project is written in Python3 and C++ and relies on TensorFlow v1.3. We have only tested it on Ubuntu 14.04. If you are using other Linux versions, we suggest using Docker. CMake (version >= 2.8) is required to compile the C++ code. Install TensorFlow (GPU-enabled) by following the instructions on https://www.tensorflow.org/install/. The project requires no other Python packages.

On Ubuntu 14.04, install the required packages by
```
sudo apt-get install cmake
sudo pip install --upgrade tensorflow-gpu
```
环境:python35,tensorflow1.3(注:本人此处使用高版本的tensorflow,进行编译时,出现undefined symbol: _ZTIN10tensorflow8OpKernelE错误)

首先创建.tf文件:
下载ICDAR2015的数据,然后运行tool/create_datasets.py文件,得到data文件中的.tf文件
## Installation

The project uses `manage.py` to execute commands for compiling code and running training and testing programs. For installation, execute
```
编译
./manage.py build_op
```
in the project directory to compile the custom TensorFlow operators written in C++. To remove the compiled binaries, execute
```
清除
./manage.py clean_op
```

## Dataset Preparation

See ``tool/create_datasets.py''

## Training
训练
```
./manage.py <exp-directory> train
```
例如:python manage.py train exp/sgd finetune_ic15

## Evaluation

See ``evaluate.py''

##数据.ckpt文件来源
https://github.com/GuoLiuFang/seglink-lfs
##模型来源
https://github.com/bgshih/seglink
