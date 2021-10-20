# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn

import dill
import pandas

from config import TrainConfig
from setting import *

from src.utils import load_args

document2id, id2document = dill.load(open(DOCUMENT_ID_PATH, 'rb'))
forward_index = dill.load(open(FORWARD_INDEX_PATH, 'rb'))
inverted_index = dill.load(open(INVERTED_INDEX_PATH, 'rb'))

print(len(document2id))
print(len(id2document))
print('#' * 128)
print(len(forward_index))
print('#' * 128)
print(len(inverted_index))
