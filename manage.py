# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn

import gensim
import pandas

from config import TrainConfig
from setting import *

from src.language_model import *
from src.utils import load_args, load_stopwords
from src.dataset import CSVDataset, generate_dataloader
from gensim.similarities import Similarity

args = load_args(TrainConfig)
args.use_reference = False
dataloader = generate_dataloader(args, mode='test')
dataloader_kd = generate_dataloader(args, mode='test_kd')
dataloader_ca = generate_dataloader(args, mode='test_ca')

