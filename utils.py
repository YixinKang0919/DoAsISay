# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import os
import zipfile

import gdown
import const


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
	
