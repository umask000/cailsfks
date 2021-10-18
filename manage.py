# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn

import pandas

from config import TrainConfig
from setting import *

from src.utils import load_args

df = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0)

print(df[df.section.isna()].head(100))
