# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import os
import zipfile

import gdown
import const
from moviepy.editor import ImageSequenceClip

def make_options(pick_targets, place_targets, options_in_api_form=True, termitation_string='done()'):
	"""generate all possible options given pick and place targets in the environments"""
	options = []
	form_str = 'robot.pick_and_place({},{})' if options_in_api_form else 'Pick the {} and place it on the {}'
	for pick in pick_targets:
		for place in place_targets:
			options.append(form_str.format(pick, place))
	options.append(termitation_string)

	print(f'Considering {len(options)} options')
	return options


def output_cached_video(env, out_name='my_video'):
	not os.path.exists('outputs') and os.makedirs('outputs')
	if not env.cache_video:
		print('No cached video, abort...')
	debug_clip = ImageSequenceClip(env.cache_video, fps=25)
	debug_clip.write_videofile(f'outputs/{out_name}.mp4')
	# clear cache
	env.cache_video = []


def try_load_all_assets():
	not os.path.exists('assets') and os.makedirs('assets')
	for ast in const.ASSET_NAMES:
		if os.path.exists(const.ASSETS_PATH_DICT[ast]):
			continue
		zip_file_name = 'assets/{}.zip'.format(ast)
		gdown.download(const.GDOWN_URL_DICT[ast], output=zip_file_name)
		with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
			zip_ref.extractall('assets')


if __name__ == "__main__":
	try_load_all_assets()
	
