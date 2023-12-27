# coding=utf-8
'''
@Author: Peizhen Li
@Desc: None
'''

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class ResNetBlock(nn.Module):
    features: int
    stride: int = 1
    
    def setup(self):
        self.conv0 = nn.Conv(self.features // 4, (1, 1), (self.stride, self.stride))
        self.conv1 = nn.Conv(self.features // 4, (3, 3))
        self.conv2 = nn.Conv(self.features, (1, 1))
        self.conv3 = nn.Conv(self.features, (1, 1), (self.stride, self.stride))
    
    def __call__(self, x):
        y = self.conv0(nn.relu(x))
        y = self.conv1(nn.relu(y))
        y = self.conv2(nn.relu(y))
        if x.shape != y.shape:
            x = self.conv3(nn.relu(x))
        return x + y



class Upsample(nn.Module):
    """simple 2D 2x bilinear upsample"""

    def __call__(self, x):
        B, H, W, C = x.shape
        new_shape = (B, H * 2, W * 2, C)
        return jax.image.resize(x, new_shape, 'bilinear')
    

class ResNet(nn.Module):
    """Hourglass 53-layer ResNet with 8-stride"""
    out_dim: int

    def setup(self):
        self.dense0 = nn.Dense(8)
        self.conv0 = nn.Conv(64, (3, 3), (1, 1))
        self.block0 = ResNetBlock(64)
        self.block1 = ResNetBlock(64)
        self.block2 = ResNetBlock(128, stride=2)
        self.block3 = ResNetBlock(128)
        self.block4 = ResNetBlock(256, stride=2)
        self.block5 = ResNetBlock(256)
        self.block6 = ResNetBlock(512, stride=2)
        self.block7 = ResNetBlock(512)

        self.block8 = ResNetBlock(256)
        self.block9 = ResNetBlock(256)
        self.upsample0 = Upsample()
        self.block10 = ResNetBlock(128)
        self.block11 = ResNetBlock(128)
        self.upsample1 = Upsample()
        self.block12 = ResNetBlock(64)
        self.block13 = ResNetBlock(64)
        self.upsample2 = Upsample()
        self.block14 = ResNetBlock(16)
        self.block15 = ResNetBlock(16)
        self.conv1 = nn.Conv(self.out_dim, (3, 3))

    def __call__(self, x, text):
        
        x = self.conv0(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)


        


if __name__ == "__main__":
    resb = ResNetBlock(64)
    print(resb.features, resb.stride)

    
    

