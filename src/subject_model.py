# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import sys
if __name__ == '__main__':
    import sys
    sys.path.append('../')

import dill
import numpy

from setting import *

# 计算词频指数(TF)：do_normalize参数指是否对文档长度进行归一化（使用最高频词的词频）
def term_frequency(forward_index, token, document_id, do_normalize=True):
    return forward_index[document_id][0].get(token) / forward_index[document_id][3 if do_normalize else 2]

# 计算逆文档指数(IDF)
def inverse_document_frequency(forward_index, inverted_index, token):
    numpy.log(len(forward_index) / (len(inverted_index[token]) + 1))

# TDIDF指数计算：do_normalize参数指是否对文档长度进行归一化（使用最高频词的词频）
def tfidf(token, document_id, forward_index=None, inverted_index=None, do_normalize=True):
    if forward_index is None:
        forward_index = dill.load(open(FORWARD_INDEX_PATH, 'wb'))
    if inverted_index is None:
        inverted_index = dill.load(open(INVERTED_INDEX_PATH, 'wb'))
    tf = term_frequency(forward_index, token, document_id, do_normalize=do_normalize)
    idf = inverse_document_frequency(forward_index, inverted_index, token)
    return tf * idf

# BM25指数计算：建议事先计算好mean_document_length参数（文档平均长度）作为参数输入 ，计算该值应该挺费时间的
def bm25(token, document_id, forward_index=None, inverted_index=None, mean_document_length=None, k=2, b=.75, do_normalize=True):
    if forward_index is None:
        forward_index = dill.load(open(FORWARD_INDEX_PATH, 'wb'))
    if inverted_index is None:
        inverted_index = dill.load(open(INVERTED_INDEX_PATH, 'wb'))
    if mean_document_length is None:
        mean_document_length = numpy.mean(list(map(lambda x: x[3 if do_normalize else 2], list(forward_index.values()))))
    tf = term_frequency(forward_index, token, document_id, do_normalize=do_normalize)
    tf_bm25 = (k + 1) * tf / (k * (1. - b + b * mean_document_length) + tf)
    idf = inverse_document_frequency(forward_index, inverted_index, token)
    return tf_bm25 * idf
