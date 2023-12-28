Vision Language to Action: Simple Practise

## DoAsISay
This is tiny demo for Vision Language to Action learning dubbed DoAsISay.
Generate plan/make decision with the help of LLM (understanding high-level language instructions from user input), and then obtain low-level pick/place position through Cliport. Specifically, the 
overall pipeline can be summarized as follows:

1. Obtain formatted solutions/plans (e.g., pick {} place {}) from user input (avaible for long-horizon tasks, e.g., move all the red boxes to the corner) through **LLM**
2. Infer pick/place position using **Cliport**
3. Make environment step

## setup
**Install Anaconda or Miniconda** first, and then use conda 
to manage your python env 
### conda env
```
conda create -n doasisay python=3.9
conda activate doasisay
```

### installation

```
# basic simulation
conda install ffmpeg -c conda-forge
pip install ftfy regex tqdm fvcore imageio==2.4.1 imageio-ffmpeg==0.4.5
pip install numpy
pip install torch
pip install pybullet
pip install -U --no-cache-dir gdown --pre
pip install matplotlib
pip install moviepy 

# vision language to action
pip install git+https://github.com/openai/CLIP.git
pip install flax==0.5.3
pip install tensorflow
# pip install easydict
```

#### Modification

Current workaround for lib confliction

```
# ===pz notes start===
# 1. flax/linen/linear.py
# from jax import ShapedArray
from jax.core import ShapedArray as ShapeArray
# 2. flax/training/checkpoints.py
# from jax.experimental.gda_serialization.serialization import get_tensorstore_spec
# from jax.experimental.gda_serialization.serialization import GlobalAsyncCheckpointManager
# from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.array_serialization.serialization import get_tensorstore_spec
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from jax import Array as GlobalDeviceArray
# pz notes end
```




