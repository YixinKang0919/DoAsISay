# coding=utf-8
'''
@Author: Peizhen Li
@Desc: Text-conditioned translation-only Transporter nets
'''
# import torch
import numpy as np
# import clip
import flax
from flax.training import checkpoints

import jax
import jax.numpy as jnp
# import optax
from TransporterNets import TransporterNets
import os
import gdown

CLIPORT_CKPT_URL = 'https://drive.google.com/uc?id=1Nq0q1KbqHOA5O7aRSu4u7-u27EMMXqgP'


@jax.jit
def eval_step(params, batch):
    pick_logits, place_logits = TransporterNets().apply({'params': params}, batch['img'], batch['text'])
    return pick_logits, place_logits

def get_pretrained_optim():
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    init_img = jnp.ones((4, 224, 224, 5), jnp.float32)
    init_text = jnp.ones((4, 512), jnp.float32)
    init_pix = jnp.zeros((4, 2), np.int32)
    init_params = TransporterNets().init(key, init_img, init_text, init_pix)['params']
    optim = flax.optim.Adam(learning_rate=1e-4).create(init_params)
    # restore checkpoints
    ckpt_path = f'checkpoints/ckpt_{40000}'
    dirname = os.path.dirname(ckpt_path)
    if not os.path.exists(ckpt_path):
        not os.path.exists(dirname) and os.makedirs(dirname)
        gdown.download(url=CLIPORT_CKPT_URL, output=ckpt_path)
    optim = checkpoints.restore_checkpoint(ckpt_path, optim)
    return optim

if __name__ == "__main__":
    # import optax
    # print(optax.adam)
    optim = get_pretrained_optim()
    # ckpt_path = f'folder1/checkpoints/ckpt_{40000}'
    # print('dirname: ', os.path.dirname(ckpt_path)) # folder1/checkpoints
    # print('base_name: ', os.path.basename(ckpt_path)) # ckpt_40000

    