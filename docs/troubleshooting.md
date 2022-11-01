

## CUDA GPU memory issues
- when multiprocessing fails due to unexpected attempt to multiprocess on the GPU, python will leak memory on the GPU. Fix with the following:
```
lsof /dev/nvidia*  # lists processes using NVIDIA GPU
# find the pid of user python
kill -9 <pid>
```
References for the above
- https://stackoverflow.com/questions/4354257/can-i-stop-all-processes-using-cuda-in-linux-without-rebooting
- https://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf