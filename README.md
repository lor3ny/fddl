# Profiling Usage

DL models are executed over single and multi GPUs systems.

# Dependencies

```cmd
module load python
module load anaconda
module load openmpi/{version-cudaaware}
module load cuda/{version}
moudle load nccl/{version} #compatible with cuda/{version}
``` 

```cmd
install python 3.x

pip install numpy
pip install matplotlib
pip install nvtx
pip install torch
pip install torchvision
pip install mpi4pi # It has to be compiled with the backend that you have (eg. Open MPI)
// Install pytorch following PyTorch settings
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
