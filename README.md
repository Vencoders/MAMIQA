# MAMIQA

This repository contains the implementation for the paper: 

MAMIQA: No-Reference Image Quality Assessment based on Multiscale Attention Mechanism with Natural Scene Statistics

## **Environment**

The proposed MAMIQA is implemented with Pytorch1.7.1 CUDA10.1 and trained under Ubuntu 16.04 operation system with TITAN RTX GPU.

**Dependencies：**

- opencv-python 4.7.0.68
- scipy 1.5.2
- openpyxl 3.0.5
- tqdm 4.50.2

## Datasets

We used five commonly used IQA datasets in this work, including three synthetic distortion datasets  ([LIVE](https://live.ece.utexas.edu/research/quality/subjective.htm), [CSIQ](https://s2.smu.edu/~eclarson/csiq.html), [TID2013](http://www.ponomarenko.info/tid2013.htm)) and two authentic distortion datasets ([CLIVE](https://live.ece.utexas.edu/research/ChallengeDB/), [KonIQ](http://database.mmsp-kn.de/koniq-10k-database.html)).

## **Training & Testing on IQA datasets**

An example for Training and testing our model :

```python
python run.py  --batch_size 64 --svpath '/home/mamiqa/Save_MAM/' --droplr 1 --epochs 5 --gpunum '0' --datapath '/tid2013' --dataset 'tid2013' --seed 1 --vesion 1
```

Some available options:

- `--dataset`: Training and testing dataset, support datasets: clive | koniq | live | csiq | tid2013
- `-train_patch_num`: Number of sample patches from training image.
- `-test_patch_num`: Number of sample patches from testing image.
- `-batch_size`: Batch size.
- `--datapath`: The path of the dataset.
- `--gpunum`: the id for the gpu that will be used.
- `--svpath`: the path to save the info.

## Models

link：https://pan.baidu.com/s/19Wwxl6_i9iF2-ZBp3sFV9g?pwd=i9iz

## **Acknowledgement**

The code of this work is adapted from **[Visual Attention Network (VAN)](https://github.com/Visual-Attention-Network/VAN-Classification/tree/main) and [TReS](https://github.com/isalirezag/TReS). Thanks for their authors.**