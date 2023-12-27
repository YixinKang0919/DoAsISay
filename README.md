Vision Language to Action: Simple Practise

## DoAsISay
This is tiny demo for Vision Language to Action learning dubbed DoAsISay.

## setup
**Install Anaconda or Miniconda** first, and then use conda 
to manage your python env 
### conda env
```
conda create -n doasisay python=3.8
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
pip install jedi
pip install flax==0.5.3
pip install easydict
```




