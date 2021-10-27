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

from gensim import corpora, models
from gensim.similarities import Similarity
from torch.utils.data import Dataset, DataLoader

from setting import *
from config import TrainConfig
from src.dataset import json_to_csv, token2frequency_to_csv, token2id_to_csv, reference_to_csv, index_reference
from src.language_model import generate_reference_corpus, build_tfidf_model, build_lsi_model, build_lda_model, query
from src.utils import load_args, load_stopwords, filter_stopwords

if __name__ == '__main__':
	os.makedirs(NEWDATA_DIR, exist_ok=True)

	# 训练集与测试集的预处理
	# token2frequency = {}
	# for filename in os.listdir(RAWDATA_DIR):
	# 	_, token2frequency = json_to_csv(import_path=os.path.join(RAWDATA_DIR, filename),
	# 									  export_path=os.path.join(NEWDATA_DIR, '.'.join(filename.split('.')[:-1]) + '.csv'),
	# 									  token2frequency=token2frequency,
	# 									  mode='train' if 'train' in filename else 'test')
	# token2frequency_to_csv(export_path=TOKEN2FREQUENCY_PATH, token2frequency=token2frequency)
	# token2id_to_csv(export_path=TOKEN2ID_PATH, token2frequency=token2frequency)

	# 参考书目的预处理
	# _, token2frequency = reference_to_csv(export_path=REFERENCE_PATH)
	# token2frequency_to_csv(export_path=REFERENCE_TOKEN2FREQUENCY_PATH, token2frequency=token2frequency)
	# token2id_to_csv(export_path=REFERENCE_TOKEN2ID_PATH, token2frequency=token2frequency)
	# index_reference()

	# 语言模型预构建
	args2 = load_args(TrainConfig)
	stopwords = load_stopwords()
	for num_best in [32]:
		for num_topic in [64]:
			for filter_stopword in [True, False]:
				args2.lsi_num_best = num_best
				args2.lda_num_best = num_best
				args2.lsi_num_topics = num_topic
				args2.lda_num_topics = num_topic
				args2.filter_stopword = filter_stopword
				with open(f'temp/query_tfidf_{args2.lsi_num_best}_{args2.lsi_num_topics}_{args2.filter_stopword}.txt', 'w', encoding='utf8') as f:
					pass
				with open(f'temp/query_lsi_{args2.lsi_num_best}_{args2.lsi_num_topics}_{args2.filter_stopword}.txt', 'w', encoding='utf8') as f:
					pass
				with open(f'temp/query_lda_{args2.lda_num_best}_{args2.lda_num_topics}_{args2.filter_stopword}.txt', 'w', encoding='utf8') as f:
					pass
				generate_reference_corpus(args2)
				build_tfidf_model()
				tfidf_model = models.TfidfModel.load(REFERENCE_TFIDF_MODEL_PATH)  # 导入TFIDF模型
				build_lsi_model(args=args2, corpus_path=REFERENCE_CORPUS_PATH)
				build_lda_model(args=args2, corpus_path=REFERENCE_CORPUS_PATH)
				corpus_tfidf = corpora.MmCorpus(REFERENCE_CORPUS_TFIDF_PATH)  						# 读取TFIDF语料
				corpus_lsi = corpora.MmCorpus(REFERENCE_CORPUS_LSI_PATH)  							# 读取LSI语料
				corpus_lda = corpora.MmCorpus(REFERENCE_CORPUS_LDA_PATH)  							# 读取LDA语料
				dictionary = corpora.Dictionary.load(REFERENCE_DICTIONARY_PATH)  					# 读取字典（即token2id信息）
				lsi_model = models.LsiModel.load(REFERENCE_LSI_MODEL_PATH)  						# 导入LSI模型
				lda_model = models.LsiModel.load(REFERENCE_LDA_MODEL_PATH)  						# 导入LDA模型
				tfidf_model = models.TfidfModel.load(REFERENCE_TFIDF_MODEL_PATH)  					# 导入TFIDF模型
				similarity_tfidf = Similarity('gensim_similarity_tfidf', corpus_tfidf, num_features=len(dictionary), num_best=args2.lsi_num_best)
				similarity_lsi = Similarity('gensim_similarity_lsi', corpus_lsi, num_features=len(dictionary), num_best=args2.lsi_num_best)
				similarity_lda = Similarity('gensim_similarity_lda', corpus_lda, num_features=len(dictionary), num_best=args2.lda_num_best)
				train_df = pandas.concat([pandas.read_csv(filepath, sep='\t', header=0) for filepath in TRAINSET_PATHs]).reset_index(drop=True)
				print(train_df.shape[0])
				for i in range(0, train_df.shape[0]):
					# print(i)
					_id = train_df.loc[i, 'id']
					statement = eval(train_df.loc[i, 'statement'])
					if args2.filter_stopword:
						statement = filter_stopwords(statement, stopwords)
					# print(type(statement))
					option_a = eval(train_df.loc[i, 'option_a'])
					option_b = eval(train_df.loc[i, 'option_b'])
					option_c = eval(train_df.loc[i, 'option_c'])
					option_d = eval(train_df.loc[i, 'option_d'])
					# type = train_df.loc[i, 'type']
					subject = train_df.loc[i, 'subject']
					# answer = train_df.loc[i, 'answer']

					# print(subject)
					# print(''.join(statement))

					# query_tokens = statement
					query_tokens = statement + option_a + option_b + option_c + option_d

					result_tfidf = query(query_tokens, dictionary, similarity_tfidf, model_sequence=[tfidf_model])
					# result_lsi = query(statement, dictionary, similarity_lsi, model_sequence=[tfidf_model, lsi_model])
					result_lsi = query(query_tokens, dictionary, similarity_lsi, model_sequence=[lsi_model])
					# result_lda = query(statement, dictionary, similarity_lda, model_sequence=[tfidf_model, lda_model])
					result_lda = query(query_tokens, dictionary, similarity_lda, model_sequence=[lda_model])
					with open(f'temp/query_tfidf_{args2.lsi_num_best}_{args2.lsi_num_topics}_{args2.filter_stopword}.txt', 'a', encoding='utf8') as f:
						f.write(f'{_id}\t{subject}\t{result_tfidf}\n')

					with open(f'temp/query_lsi_{args2.lsi_num_best}_{args2.lsi_num_topics}_{args2.filter_stopword}.txt', 'a', encoding='utf8') as f:
						f.write(f'{_id}\t{subject}\t{result_lsi}\n')

					with open(f'temp/query_lda_{args2.lda_num_best}_{args2.lda_num_topics}_{args2.filter_stopword}.txt', 'a', encoding='utf8') as f:
						f.write(f'{_id}\t{subject}\t{result_lda}\n')
