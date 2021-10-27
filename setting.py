# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import os

# ----
DATA_DIR = 'data'
RAWDATA_DIR = os.path.join(DATA_DIR, 'raw')
NEWDATA_DIR = os.path.join(DATA_DIR, 'new')
REFERENCE_DIR = os.path.join(DATA_DIR, 'reference_book')
STOPWORDS_DIR = os.path.join(DATA_DIR, 'stopwords-master')

TRAINSET_PATHs = [os.path.join(NEWDATA_DIR, '0_train.csv'), os.path.join(NEWDATA_DIR, '1_train.csv')]
TESTSET_PATHs = [os.path.join(NEWDATA_DIR, '0_test.csv'), os.path.join(NEWDATA_DIR, '1_test.csv')]

TOKEN2ID_PATH = os.path.join(NEWDATA_DIR, 'token2id.csv')
TOKEN2FREQUENCY_PATH = os.path.join(NEWDATA_DIR, 'token2frequency.csv')
REFERENCE_PATH = os.path.join(NEWDATA_DIR, 'reference_book.csv')
COMPLETE_REFERENCE_PATH = os.path.join(NEWDATA_DIR, 'complete_reference_book.csv')
REFERENCE_TOKEN2ID_PATH = os.path.join(NEWDATA_DIR, 'reference_token2id.csv')
REFERENCE_TOKEN2FREQUENCY_PATH = os.path.join(NEWDATA_DIR, 'reference_token2frequency.csv')

STOPWORD_PATHs = {
    'baidu': os.path.join(STOPWORDS_DIR, 'baidu_stopwords.txt'),
    'cn': os.path.join(STOPWORDS_DIR, 'cn_stopwords.txt'),
    'hit': os.path.join(STOPWORDS_DIR, 'hit_stopwords.txt'),
    'scu': os.path.join(STOPWORDS_DIR, 'scu_stopwords.txt'),
}

# 这三个.dat文件自从20211021使用gensim生成语言模型后大概率是要弃用的了
DOCUMENT_ID_PATH = os.path.join(NEWDATA_DIR, 'document_id.dat')
FORWARD_INDEX_PATH = os.path.join(NEWDATA_DIR, 'forward_index.dat')
INVERTED_INDEX_PATH = os.path.join(NEWDATA_DIR, 'inverted_index.dat')

# ----
LOGGING_DIR = 'logging'

TENSORBOARD_DIR = os.path.join(LOGGING_DIR, 'tensorboard')

# ----
CHECKPOINT_DIR = 'checkpoint'

# ----
MODEL_DIR = 'model'
REFERENCE_DICTIONARY_PATH = os.path.join(MODEL_DIR, 'reference_dictionary.dat')
REFERENCE_CORPUS_PATH = os.path.join(MODEL_DIR, 'reference_corpus.dat')
REFERENCE_CORPUS_TFIDF_PATH = os.path.join(MODEL_DIR, 'reference_corpus_tfidf.dat')
REFERENCE_CORPUS_LSI_PATH = os.path.join(MODEL_DIR, 'reference_corpus_lsi.dat')
REFERENCE_CORPUS_LDA_PATH = os.path.join(MODEL_DIR, 'reference_corpus_lda.dat')
REFERENCE_TFIDF_MODEL_PATH = os.path.join(MODEL_DIR, 'reference_tfidf.m')
REFERENCE_LSI_MODEL_PATH = os.path.join(MODEL_DIR, 'reference_lsi.m')
REFERENCE_LDA_MODEL_PATH = os.path.join(MODEL_DIR, 'reference_lda.m')

OPTION2INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
INDEX2OPTION = {index: option for option, index in OPTION2INDEX.items()}
FREQUENCY_THRESHOLD = 1
TOKEN2ID = {'PAD': 0, 'UNK': 1}
SUBJECT2INDEX = {'法制史' if subject == '目录和中国法律史' else subject: index + 1 for index, subject in enumerate(os.listdir(REFERENCE_DIR))}
INDEX2SUBJECT = {index: subject for subject, index in SUBJECT2INDEX.items()}

# ----
# 预训练模型
PRETRAINED_MODELS = {

}
