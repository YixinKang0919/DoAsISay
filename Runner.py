# coding=utf-8
'''
@Author: Peizhen Li
@Desc: None
'''

import torch
import numpy as np
from PickPlaceEnv import env
import clip
import jax.numpy as jnp
from Cliport.Cliport import eval_step, get_pretrained_optim
import matplotlib.pyplot as plt
from utils import output_cached_video 


def get_pretrained_clip():
    clip_model, _ = clip.load('ViT-B/32')
    clip_model.eval() # or clip_model.cuda().eval() if cuda is available
    return clip_model

def get_coords():
    # Coordinate map (i.e. position encoding).
    coord_x, coord_y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224), sparse=False, indexing='ij')
    coords = np.concatenate((coord_x[..., None], coord_y[..., None]), axis=2)
    return coords


def run_cliport(obs, text):
    before = env.get_camera_image() 
    pre_obs = obs['image'].copy()

    # tokenize text and get CLIP features
    text_tokens = clip.tokenize(text) # .cuda()
    clip_model = get_pretrained_clip()
    with torch.no_grad():
        text_feats = clip_model.encode_text(text_tokens).float()
    text_feats /= text_feats.norm(dim=-1, keepdim=True)
    text_feats = np.float32(text_feats.cpu())

    # normalize image and add batch dimension
    coords = get_coords()
    img = obs['image'][None, ...] / 255
    img = np.concatenate((img, coords[None, ...]), axis=3)

    # run TransporterNets to get pick and place heatmaps
    batch = {'img': jnp.float32(img), 'text': jnp.float32(text_feats)}
    optim = get_pretrained_optim()
    pick_map, place_map = eval_step(optim.target, batch)
    pick_map, place_map = np.float32(pick_map), np.float32(place_map)

    # get pick position
    pick_max = np.argmax(np.float32(pick_map)).squeeze()
    pick_yx = (pick_map // 224, pick_max % 224) # (row, col)
    pick_yx = np.clip(pick_yx, 20, 204)
    pick_xyz = obs['xyzmap'][pick_yx[0], pick_yx[1]]

    # get place position
    place_max = np.argmax(np.float32(place_map)).squeeze()
    place_yx = (place_max // 224, place_max % 224)
    place_yx = np.clip(place_yx, 20, 204)
    place_xyz = obs['xyzmap'][place_yx[0], place_yx[1]]

    # step environment
    act = {'pick': pick_xyz, 'place': place_xyz}
    obs, _, _, _ = env.step(act)

    # show pick and place action
    plt.title(text)
    plt.imshow(pre_obs)
    plt.arrow(pick_yx[1], pick_yx[0], place_yx[1]-pick_yx[1], place_yx[0]-pick_yx[0], 
              color='w', head_starts_at_zero=False, head_width=7, length_includes_head=True)
    plt.show()

    plt.subplot(1, 2, 1)
    plt.title('Pick Heatmap')
    plt.imshow(pick_map.reshape(224, 224))
    plt.subplot(1, 2, 2)
    plt.title('Place Heatmap')
    plt.imshow(place_map.reshape(224, 224))
    plt.show()

    # save environment rollout as video
    output_cached_video(env)

    # show camera image before and after
    plt.subplot(1, 2, 1)
    plt.title('Before')
    plt.imshow(before)
    plt.subplot(1, 2, 2)
    plt.title('After')
    plt.imshow(env.get_camera_image())
    plt.show()

    return obs

    




