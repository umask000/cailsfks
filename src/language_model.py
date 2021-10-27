# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 语言模型：主要用于文档检索

import sys
if __name__ == '__main__':
    import sys
    sys.path.append('../')

import time
import dill
import numpy
import jieba
import pandas

from gensim import corpora, models
from gensim.similarities import Similarity

from setting import *
from src.utils import load_stopwords, filter_stopwords


# 计算词频指数(TF)：do_normalize参数指是否对文档长度进行归一化（使用最高频词的词频）
def term_frequency(forward_index, token, document_id, do_normalize=True):
    return forward_index[document_id][0].get(token) / forward_index[document_id][3 if do_normalize else 2]

# 计算逆文档指数(IDF)
def inverse_document_frequency(forward_index, inverted_index, token):
    return numpy.log(len(forward_index) / (len(inverted_index[token]) + 1.))

# TDIDF指数计算：do_normalize参数指是否对文档长度进行归一化（使用最高频词的词频）
def tfidf(token, document_id, forward_index=None, inverted_index=None, do_normalize=True):
    if forward_index is None:
        forward_index = dill.load(open(FORWARD_INDEX_PATH, 'wb'))
    if inverted_index is None:
        inverted_index = dill.load(open(INVERTED_INDEX_PATH, 'wb'))
    tf = term_frequency(forward_index, token, document_id, do_normalize=do_normalize)
    idf = inverse_document_frequency(forward_index, inverted_index, token)
    return tf * idf

# 改进的TDIDF指数计算：Lucene中的改进做法，我理解只是把TF指数开了个根号
def tfidf_lucene(token, document_id, forward_index=None, inverted_index=None, do_normalize=True):
    if forward_index is None:
        forward_index = dill.load(open(FORWARD_INDEX_PATH, 'wb'))
    if inverted_index is None:
        inverted_index = dill.load(open(INVERTED_INDEX_PATH, 'wb'))
    tf = numpy.sqrt(term_frequency(forward_index, token, document_id, do_normalize=do_normalize))
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
    # tf = term_frequency(forward_index, token, document_id, do_normalize=do_normalize)
    tf = forward_index[document_id][0].get(token)
    L = forward_index[document_id][2] / mean_document_length
    tf_bm25 = (k + 1.) * tf / (k * (1. + b * (L - 1.)) + tf)
    idf = inverse_document_frequency(forward_index, inverted_index, token)
    return tf_bm25 * idf

# ----
# 20211020: 以上为手动实现的TFIDF和BM25算法，后来发现可以直接使用Gensim库中相关方法，因而废弃且数据预处理中的index_reference也无用

# 生成参考书目语料：即gensim框架下的词频信息与字典
def generate_reference_corpus(args):
    if args.filter_stopword:
        stopwords = load_stopwords()

    reference_dataframe = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0, dtype=str)
    reference_dataframe = reference_dataframe.fillna('')                # section字段存在缺失，使用空字符串填充
    document = []
    for i in range(reference_dataframe.shape[0]):
        # law = reference_dataframe.loc[i, 'law']
        # chapter_name = reference_dataframe.loc[i, 'chapter_name']
        section = reference_dataframe.loc[i, 'section']
        content = eval(reference_dataframe.loc[i, 'content'])

        # paragraph = content[:]                                          # 不把章节名称作为文档段落
        paragraph = jieba.lcut(section) + content                       # 把章节名称作为文档段落：因为我把章节名称从文档段落中分离出来了
        if args.filter_stopword:
            paragraph = filter_stopwords(tokens=paragraph, stopwords=stopwords)
        document.append(paragraph)

    dictionary = corpora.Dictionary(document)                           # 生成文档字典
    corpus = [dictionary.doc2bow(paragraph) for paragraph in document]  # 生成语料：权重是词频信息
    dictionary.save(REFERENCE_DICTIONARY_PATH)                          # 保存生成的字典
    corpora.MmCorpus.serialize(REFERENCE_CORPUS_PATH, corpus)           # 保存生成的语料
    return corpus, dictionary

# 构建TFIDF模型并利用TFIDF模型生成新的TFIDF指数矩阵
def build_tfidf_model(corpus_path=REFERENCE_CORPUS_PATH, dictionary_path=REFERENCE_DICTIONARY_PATH):
    corpus = corpora.MmCorpus(corpus_path)
    dictionary = corpora.Dictionary.load(dictionary_path)                   # 读取字典（即token2id信息）
    tfidf_model = models.TfidfModel(corpus)                                 # 生成tfidf模型
    corpus_tfidf = [tfidf_model[doc] for doc in corpus]                     # 转换为tfidf指数矩阵语料
    tfidf_model.save(REFERENCE_TFIDF_MODEL_PATH)                            # 保存tfidf模型
    corpora.MmCorpus.serialize(REFERENCE_CORPUS_TFIDF_PATH, corpus_tfidf)   # 保存tfidf指数矩阵语料
    return tfidf_model, corpus_tfidf


# 构建LSI模型并利用LSI模型生成新的LSI指数矩阵
def build_lsi_model(args, corpus_path=REFERENCE_CORPUS_TFIDF_PATH, dictionary_path=REFERENCE_DICTIONARY_PATH):
    corpus = corpora.MmCorpus(corpus_path)                              # 读取语料（这里的语料目前是tfidf指数矩阵）
    dictionary = corpora.Dictionary.load(dictionary_path)               # 读取字典（即token2id信息）
    lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=args.lsi_num_topics)
    lsi_model.save(REFERENCE_LSI_MODEL_PATH)                            # 保存LSI模型
    corpus_lsi = [lsi_model[doc] for doc in corpus]                     # 转换为LSI指数矩阵语料
    corpora.MmCorpus.serialize(REFERENCE_CORPUS_LSI_PATH, corpus_lsi)   # 保存LSI指数矩阵语料
    return lsi_model, corpus_lsi

# 构建LDA模型并利用LDA模型生成新的LDA指数矩阵
def build_lda_model(args, corpus_path=REFERENCE_CORPUS_TFIDF_PATH, dictionary_path=REFERENCE_DICTIONARY_PATH):
    corpus = corpora.MmCorpus(corpus_path)                              # 读取语料（这里的语料目前是tfidf指数矩阵）
    dictionary = corpora.Dictionary.load(dictionary_path)               # 读取字典（即token2id信息）
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=args.lda_num_topics)
    lda_model.save(REFERENCE_LDA_MODEL_PATH)                            # 保存lda模型
    corpus_lda = [lda_model[doc] for doc in corpus]                     # 转换为LDA指数矩阵语料
    corpora.MmCorpus.serialize(REFERENCE_CORPUS_LDA_PATH, corpus_lda)   # 保存LDA指数矩阵语料
    return lda_model, corpus_lda

# 查询检索
def query(query_tokens, dictionary, similarity, model_sequence):
    query_corpus = dictionary.doc2bow(query_tokens)
    for model in model_sequence:
        query_corpus = model[query_corpus]
    return similarity[query_corpus]

# 基于gensim文本匹配模型的文档检索函数
def retrieve_paragraphs(reference, query_tokens, dictionary, similarity, model_sequence):
    result = query(query_tokens, dictionary, similarity, model_sequence)
    paragraphs = []
    for (index, score) in result:
        paragraph = eval(reference.loc[:, 'content'])
        paragraphs.append(paragraph)
    return paragraphs, result

# 更新语言模型
def update_language_model(args):
    generate_reference_corpus(args)
    build_tfidf_model()
    # build_lsi_model(args=args, corpus_path=REFERENCE_CORPUS_PATH)
    # build_lda_model(args=args, corpus_path=REFERENCE_CORPUS_PATH)
    build_lsi_model(args=args)
    build_lda_model(args=args)

# 加载指数语料数据
def load_corpus(corpus_name):
    # 读取TFIDF语料
    if corpus_name == 'tfidf':
        corpus = corpora.MmCorpus(REFERENCE_CORPUS_TFIDF_PATH)
    # 读取LSI语料
    elif corpus_name == 'lsi':
        corpus = corpora.MmCorpus(REFERENCE_CORPUS_LSI_PATH)
    # 读取LDA语料
    elif corpus_name == 'lda':
        corpus = corpora.MmCorpus(REFERENCE_CORPUS_LDA_PATH)
    return corpus

# 加载语言模型：目前以较硬的代码来实现，可能之后会用更先进的模型
def load_language_model_sequence(model_names):
    model_sequence = []
    for model_name in model_names:
        if model_name == 'tfidf':
            model = models.TfidfModel.load(REFERENCE_TFIDF_MODEL_PATH)
        elif model_name == 'lsi':
            model = models.LsiModel.load(REFERENCE_LSI_MODEL_PATH)
        elif model_name == 'lda':
            model = models.LdaModel.load(REFERENCE_LDA_MODEL_PATH)
        else:
            assert False, f'Unknown model name: {model_name}'
        model_sequence.append(model)
    return model_sequence

def generate_similarity(args, corpus_name):
    if corpus_name == 'tfidf':
        num_best = args.lsi_num_best
    elif corpus_name == 'lsi':
        num_best = args.lsi_num_best
    elif corpus_name == 'lda':
        num_best = args.lda_num_best
    dictionary = corpora.Dictionary.load(REFERENCE_DICTIONARY_PATH)
    similarity = Similarity('gensim_similarity', load_corpus(corpus_name), num_features=len(dictionary), num_best=num_best)
    return similarity
