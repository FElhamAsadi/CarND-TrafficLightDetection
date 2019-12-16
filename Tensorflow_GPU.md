## Making Tensorflow-GPU to work!!
- Step 1: Check the software you will need to install
Assuming that Windows is already installed on your PC, the additional bits of software you will install as part of these steps are: Microsoft Visual Studio, the NVIDIA CUDA Toolkit, NVIDIA cuDNN
Python, Tensorflow (with GPU support)

- Step 2: NVIDIA GPU and Driver Installation: [Find your gpu model here, download and install it](https://www.nvidia.com/Download/index.aspx?lang=en-us). 


- Step 3: Download Visual Studio Express 
Visual Studio is a Prerequisite for CUDA Toolkit


- Step 4: Download and install CUDA Toolkit for Windows 10
At the time of writing, the default version of CUDA Toolkit offered is version 10.1. However, check which version of CUDA Toolkit you choose for download and installation to ensure compatibility with Tensorflow (it seems tensorflow works with version 10.0 not 10.1). When you go onto the Tensorflow website, the latest version of Tensorflow available (1.13.0) requires CUDA 10.0, not CUDA 10.1. To find CUDA 10.0, you need to navigate to the “Legacy Releases” on the bottom right hand side.

- Step 5: Download and Install cuDNN
Find a compatible version of CuDNN from: https://developer.nvidia.com/cudnn
you should register in nvidia first. Based on the information on the Tensorflow website, Tensorflow with GPU support requires a cuDNN version of at least 7.2. Then, unzipping cuDNN files and copying to CUDA folders: 
  * Copy cudnn64_7.dll file (can be found in the ...\cuda\bin\cudnn64_7.dll) into C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin\ 
  * Copy cudnn.h file (can be found in the ...\cuda\ include\cudnn.h) into C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include\
  * Copy cudnn.lib file (can be found in the ...\cuda\lib\x64\cudnn.lib) into C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64\

- Step 6: Run cmd as an administrator
```
C:\> SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin;%PATH%
C:\>SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64;%PATH%
C:\>SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include;%PATH%
C:\>SET PATH=C:\tools\cuda\bin;%PATH%
```
- Step 7: Install Python (if you don’t already have it)

- Step 8: Install Tensorflow with GPU support (cmd administrator)
```
pip3 install --upgrade tensorflow-gpu
```

- Step 9: In one of anaconda's IDE check if tf has been installed:
```
import tensorflow as tf 
tf.test.is_built_with_cuda()
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
```
if you get True answer, it means it works!

