# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import os
import zipfile

import gdown
import const
from moviepy.editor import ImageSequenceClip


def output_cached_video(env, out_path='outputs/my_video.mp4'):
	if not env.cache_video:
		print('No cached video, abort...')
	debug_clip = ImageSequenceClip(env.cache_video, fps=25)
	debug_clip.write_videofile(out_path)
	# clear cache
	env.cache_video = []


def try_load_all_assets():
	for ast in const.ASSET_NAMES:
		if os.path.exists(const.ASSETS_PATH_DICT[ast]):
			continue
		zip_file_name = 'assets/{}.zip'.format(ast)
		gdown.download(const.GDOWN_URL_DICT[ast], output=zip_file_name)
		with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
			zip_ref.extractall('assets')


if __name__ == "__main__":
	try_load_all_assets()
	
