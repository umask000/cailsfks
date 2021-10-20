# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import os

DATA_DIR = 'data'
RAWDATA_DIR = os.path.join(DATA_DIR, 'raw')
NEWDATA_DIR = os.path.join(DATA_DIR, 'new')
REFERENCE_DIR = os.path.join(DATA_DIR, 'reference_book')

TRAINSET_PATHs = [os.path.join(NEWDATA_DIR, '0_train.csv'), os.path.join(NEWDATA_DIR, '1_train.csv')]
TESTSET_PATHs = [os.path.join(NEWDATA_DIR, '0_test.csv'), os.path.join(NEWDATA_DIR, '1_test.csv')]

TOKEN2ID_PATH = os.path.join(NEWDATA_DIR, 'token2id.csv')
TOKEN2FREQUENCY_PATH = os.path.join(NEWDATA_DIR, 'token2frequency.csv')
REFERENCE_PATH = os.path.join(NEWDATA_DIR, 'reference_book.csv')
REFERENCE_TOKEN2ID_PATH = os.path.join(NEWDATA_DIR, 'reference_token2id.csv')
REFERENCE_TOKEN2FREQUENCY_PATH = os.path.join(NEWDATA_DIR, 'reference_token2frequency.csv')

DOCUMENT_ID_PATH = os.path.join(NEWDATA_DIR, 'document_id.dat')
FORWARD_INDEX_PATH = os.path.join(NEWDATA_DIR, 'forward_index.dat')
INVERTED_INDEX_PATH = os.path.join(NEWDATA_DIR, 'inverted_index.dat')

LOGGING_DIR = 'logging'

TENSORBOARD_DIR = os.path.join(LOGGING_DIR, 'tensorboard')

CHECKPOINT_DIR = 'checkpoint'

OPTION2INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
INDEX2OPTION = {index: option for option, index in OPTION2INDEX.items()}
FREQUENCY_THRESHOLD = 1
TOTAL_REFERENCE_BLOCK = 6

TOKEN2ID = {'PAD': 0, 'UNK': 1}

