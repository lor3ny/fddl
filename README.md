# Custom Hybrid Distributed Deep Learning with Fine-Grained Control

## Introduction
Large language models (LLMs)havegrownsignificantly in size and computational demands, making both training and inference increasingly expensive. To meet these requirements, supercomputers have evolved to support distributed deep learning at Exascale. However, distributing deep learning models across multiple nodes re mains a complex challenge but fundamental to reduce latency and energy consumption. While many high-level APIs aim to simplify distributed training, they often lack transparency, preventing fine-grained optimization and making it difficult to support or customize hybrid parallelism strategies. As a result, diagnosing failures or addressing performance bottlenecks becomes significantly more challenging. To address these challenges, this project investigates the use of PyTorch in conjunction with MPIandNCCL.Specifically, this work presents distributed training and inference strategies using a variational autoencoder (VAE) as a baseline model on CINECA’s Leonardo supercomputer. It introduces a fine grained data parallel solution and a hybrid data–tensor parallel approach, both designed for multi-GPU, multi-node systems, highlighting the potential of custom hybrid distribution strategies.

## How to Use
DL models are executed over single and multi-GPUs systems using CINECA's Leonardo.
This project is designed to work on cluster architectures.
So the architectures and modules have to be adapted on the chosen cluster.

The project has been testes with the following modules and versions:

- Python 3.11.6
- Anaconda 3 2023.09
- Open MPI 4.1.6 without CUDA-Awareness
- CUDA 12.1 
- NCCL 2.19.1 compatible with CUDA 12.1

# Dependencies

Before running and testing it is fundamental to upload the necessary modules and create the conda environment.

```cmd
module load python
module load anaconda
module load openmpi/{version-cudaaware or not}
module load cuda/{version}
moudle load nccl/{version} #compatible with cuda/{version}
``` 

```cmd
install python 3.x

conda create {env-name}

conda activate {env-name}

pip install numpy
pip install matplotlib
pip install nvtx
pip install torch
pip install torchvision
pip install mpi4pi # It has to be compiled with the backend that you have (eg. Open MPI)
// Install pytorch following PyTorch settings
```

To install PyTorch verify the CUDA availability on your cluster, and then choose the compatible PyTorch package.

```cmd
https://pytorch.org/get-started/locally/
```

Check CUDA compatibility [PyTorch Get Started](https://pytorch.org/get-started/locally/)

# Nsight profilee

1. Execute traingin with NVIDIA Nsight enabled

```cmd
nsys profile --trace=cuda,nvtx --stats=true --output=autoencoder_prof python train.py
```

2. Profile using the UI

```cmd
nsys-ui profile_name.nsys-prep
```
