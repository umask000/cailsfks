# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# Data preprocess

import os
import time
import json
import jieba
import torch
import pandas

from torch.utils.data import Dataset, DataLoader

from setting import *
from src.dataset import json_to_csv, token2frequency_to_csv, token2id_to_csv, reference_to_csv, index_reference

if __name__ == '__main__':
	os.makedirs(NEWDATA_DIR, exist_ok=True)
	# token2frequency = {}
	# for filename in os.listdir(RAWDATA_DIR):
	# 	_, token2frequency = json_to_csv(import_path=os.path.join(RAWDATA_DIR, filename),
	# 									  export_path=os.path.join(NEWDATA_DIR, '.'.join(filename.split('.')[:-1]) + '.csv'),
	# 									  token2frequency=token2frequency,
	# 									  mode='train' if 'train' in filename else 'test')
	# token2frequency_to_csv(export_path=TOKEN2FREQUENCY_PATH, token2frequency=token2frequency)
	# token2id_to_csv(export_path=TOKEN2ID_PATH, token2frequency=token2frequency)

	# _, token2frequency = reference_to_csv(export_path=REFERENCE_PATH)
	# token2frequency_to_csv(export_path=REFERENCE_TOKEN2FREQUENCY_PATH, token2frequency=token2frequency)
	# token2id_to_csv(export_path=REFERENCE_TOKEN2ID_PATH, token2frequency=token2frequency)
	index_reference()