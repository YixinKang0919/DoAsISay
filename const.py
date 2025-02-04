# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None

import numpy as np

# assets names
BODY = 'body'
ROBOT = 'robot'
BOWL = 'bowl'

ASSET_NAMES = set([BODY, ROBOT, BOWL])
# ================

# download url
GDOWN_URL_PREFIX = 'https://drive.google.com/uc?id='
GDOWN_URL_DICT = {
	BODY: GDOWN_URL_PREFIX+'1yOMEm-Zp_DL3nItG9RozPeJAmeOldekX',
	ROBOT: GDOWN_URL_PREFIX+'1Cc_fDSBL6QiDvNT4dpfAEbhbALSVoWcc',
	BOWL: GDOWN_URL_PREFIX+'1GsqNLhEl9dd4Mc3BM0dX3MibOI1FVWNM',}

# assets path
ASSETS_PATH_DICT = {
	BODY: 'assets/robotiq_2f_85/robotiq_2f_85.urdf',
	ROBOT: 'assets/ur5e/ur5e.urdf',
	BOWL: 'assets/bowl/bowl.urdf'}

PLANE_URDF = 'plane.urdf'

# =========
# workspace parameters
PIXEL_SIZE = 0.00267857
BOUNDS = np.float32([[-0.3, 0.3], [-0.8, -0.2], [0, 0.15]]) # X Y Z

COLORS = {
    'blue':(78/255, 121/255, 167/255, 255/255),
    'red':(255/255, 87/255, 89/255, 255/255),
    'green': (89/255, 169/255, 79/255, 255/255),
    'yellow': (237/255, 201/255, 72/255, 255/255),
    'purple': (148/255, 87/255, 235/255, 255/255),
    'orange': (255/255, 165/255, 0/255, 255/255),
    'cyan': (0/255, 255/255, 255/255, 200/255),
    'pink': (255/255, 182/255, 193/255, 180/255)
}

# default pick and place targets
PICK_TARGETS = {
    'blue block': None,
    'red block': None,
    'green block': None,
    'yellow block': None,
}

PLACE_TARGETS = {
    'blue block': None,
    'red block': None,
    'green block': None,
    'yellow block': None,

    'blue bowl': None,
    'red bowl': None,
    'green bowl': None,
    'yellow bowl': None,
    # (x, y, z)
    'top left corner': (-0.3 + 0.05, -0.2 - 0.05, 0),
    'top right corner': (0.3 + 0.05, -0.2 - 0.05, 0),
    'middle': (0, -0.5, 0),
    'bottom left corner': (-0.3 + 0.05, -0.8 + 0.05, 0),
    'bottom right corner': (0.3 - 0.05, -0.8 + 0.05, 0),
}


