# Profiling Usage

DL models are executed over single and multi GPUs systems.

# Dependencies

```cmd
install python 3.x

pip install numpy
pip install matplotlib
pip install nvtx
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