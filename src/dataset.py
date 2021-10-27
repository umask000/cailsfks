# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# Data preprocess

import sys

if __name__ == '__main__':
    import sys

    sys.path.append('../')

import os
import time
import json
import dill
import jieba
import torch
import pandas
import pickle
import networkx

from setting import *
from config import TrainConfig
from src.utils import load_args, load_stopwords, encode_answer, chinese_to_number, filter_stopwords
from src.language_model import retrieve_paragraphs, load_language_model_sequence, generate_similarity, query

from collections import Counter
from gensim import corpora, models
from torch.utils.data import Dataset, DataLoader

def tokenize(sentence, token2frequency):
    tokens = []
    for token in jieba.cut(sentence):
        tokens.append(token)
        if not token in token2frequency:
            token2frequency[token] = 0
        token2frequency[token] += 1
    return tokens, token2frequency

# JEC-QA数据集中的两组训练集和测试集从JSON格式转为CSV格式文件
def json_to_csv(import_path, export_path, token2frequency=None, mode='train'):
    assert mode in ['train', 'test'], f'Unknown param `mode`: {mode}'
    data_dict = {
        'id': [],
        'statement': [],
        'option_a': [],
        'option_b': [],
        'option_c': [],
        'option_d': [],
        'type': [],
        'subject': [],
    }
    _token2frequency = {} if token2frequency is None else token2frequency.copy()
    if mode == 'train':
        data_dict['answer'] = []
    with open(import_path, 'r', encoding='utf8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            data = json.loads(line)
            assert len(data['option_list']) == 4
            _id = data['id']
            statement, _token2frequency = tokenize(data['statement'], _token2frequency)
            option_a, _token2frequency = tokenize(data['option_list']['A'], _token2frequency)
            option_b, _token2frequency = tokenize(data['option_list']['B'], _token2frequency)
            option_c, _token2frequency = tokenize(data['option_list']['C'], _token2frequency)
            option_d, _token2frequency = tokenize(data['option_list']['D'], _token2frequency)
            _type = data['type']
            subject = data.get('subject')
            data_dict['id'].append(_id)
            data_dict['statement'].append(statement)
            data_dict['option_a'].append(option_a)
            data_dict['option_b'].append(option_b)
            data_dict['option_c'].append(option_c)
            data_dict['option_d'].append(option_d)
            data_dict['type'].append(_type)
            data_dict['subject'].append(subject)
            if mode == 'train':
                answer = encode_answer(data['answer'])
                data_dict['answer'].append(answer)
    dataframe = pandas.DataFrame(data_dict, columns=list(data_dict.keys()))
    if export_path is not None:
        dataframe.to_csv(export_path, sep='\t', index=False, header=True)
    return dataframe, _token2frequency

# token2id字典转为CSV格式文件
def token2id_to_csv(export_path, token2frequency):
    token2id = TOKEN2ID.copy()
    _id = len(token2id)
    for token, frequency in token2frequency.items():
        if frequency >= FREQUENCY_THRESHOLD:
            token2id[token] = _id
            _id += 1
    pandas.DataFrame({
        'id': list(token2id.values()),
        'token': list(token2id.keys()),
    }).sort_values(by=['id'], ascending=True).to_csv(export_path, sep='\t', index=False, header=True)

# token2frequency字典转为CSV格式文件
def token2frequency_to_csv(export_path, token2frequency):
    pandas.DataFrame({
        'token': list(token2frequency.keys()),
        'frequency': list(token2frequency.values()),
    }).sort_values(by=['frequency'], ascending=False).to_csv(export_path, sep='\t', index=False, header=True)

# 参考书目TXT文本转为CSV格式文件
def reference_to_csv(export_path, token2frequency=None):
    reference_dict = {
        'law': [],
        'chapter_number': [],
        'chapter_name': [],
        'section': [],
        'content': [],
    }
    _token2frequency = {} if token2frequency is None else token2frequency.copy()
    for law in os.listdir(REFERENCE_DIR):
        for filename in os.listdir(os.path.join(REFERENCE_DIR, law)):
            if filename.endswith('.txt'):
                _filename = filename.replace(' ', '').replace('.txt', '')
                start_index = _filename.find('第') + 1
                end_index = _filename.find('章')
                if start_index == 0 or end_index == -1:
                    continue
                chapter_number_1 = _filename[start_index: end_index]
                chapter_name_1 = _filename[end_index + 1: ]
                filepath = os.path.join(REFERENCE_DIR, law, filename)
                with open(filepath, 'r', encoding='utf8') as f:
                    lines = eval(f.read())
                total_lines = len(lines)
                for i in range(total_lines):
                    line_string = lines[i].replace(' ', '')
                    start_index = line_string.find('第') + 1
                    end_index = line_string.find('章')
                    chapter_number_2 = line_string[start_index: end_index]
                    if start_index != 0 and end_index != -1:
                        chapter_name_2 = lines[i + 1].replace(' ', '') if line_string[-1] == '章' else line_string[line_string.find('章') + 1:]
                        break
                chapter_number = chinese_to_number(chapter_number_1)
                chapter_name = chapter_name_1 if chapter_name_1 else chapter_name_2
                for i in range(total_lines):
                    blocks = lines[i].strip().split(' ')
                    section = ' '.join(blocks[: -1])
                    content, _token2frequency = tokenize(blocks[-1], _token2frequency)
                    reference_dict['law'].append(law)
                    reference_dict['chapter_number'].append(chapter_number)
                    reference_dict['chapter_name'].append(chapter_name)
                    reference_dict['section'].append(section)
                    reference_dict['content'].append(content)

    reference_dataframe = pandas.DataFrame(reference_dict, columns=list(reference_dict.keys()))
    if export_path is not None:
        reference_dataframe.to_csv(export_path, sep='\t', header=True, index=False)
    return reference_dataframe, _token2frequency

# 给参考书目制作文档索引：正排索引和倒排索引, 这将有利于后续的数据处理
def index_reference():
    reference_dataframe = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0)
    reference_token2id = pandas.read_csv(REFERENCE_TOKEN2ID_PATH, sep='\t', header=0)

    document2id = {}    # 字典键为文档四元组(law, chapter_number, chapter_name, section), 值为文档编号
    id2document = {}    # 字典键为文档编号, 值为文档四元组(law, chapter_number, chapter_name, section)

    forward_index = {}                                                      # 前向索引字典: 键为文档编号, 值为四元组(分词词频信息, 分词段落集合, 文档总词数, 文档最高频词的词频)
    inverted_index = {token: {} for token in reference_token2id['token']}   # 倒排索引字典: 键为分词, 值为该分词出现过的文档与在该文档中的出现次数的字典

    document_id = -1
    for document, df in reference_dataframe.groupby(['law', 'chapter_number', 'chapter_name', 'section']):
        # 将四元组(law, chapter_number, chapter_name, section)相同的内容划分为同一文档

        # 记录文档编号
        document_id += 1
        document2id[document] = document_id
        id2document[document_id] = document

        paragraphs = [] # 存储同一文档内的所有段落
        tokens = []     # 存储同一文档内的所有分词
        for content in df['content']:
            paragraphs.append(eval(content))
            tokens.extend(eval(content))
        token_counter = Counter(tokens) # 统计每个分词的出现次数

        # 构建正排索引与倒排索引
        forward_index[document_id] = (token_counter, paragraphs, len(tokens), token_counter.most_common(1)[0][1])
        for token, frequency in token_counter.items():
            inverted_index[token][document_id] = frequency

    # 导出为dill文件
    dill.dump((document2id, id2document), open(DOCUMENT_ID_PATH, 'wb'))
    dill.dump(forward_index, open(FORWARD_INDEX_PATH, 'wb'))
    dill.dump(inverted_index, open(INVERTED_INDEX_PATH, 'wb'))
    return document2id, id2document, forward_index, inverted_index

# 根据
def generate_index2subject(reference_df):
    index2subject = {index: '法制史' if law == '目录和中国法律史' else law for index, law in enumerate(reference_df['law'])}
    return index2subject

# 生成数据加载器(torch)
def generate_dataloader(args, mode='train'):
    batch_size = args.batch_size
    num_workers = args.num_workers
    csvdataset = CSVDataset(args, mode=mode)

    def _collate_fn(batch_data):
        _collate_data = {'id': [data['id'] for data in batch_data],
                         'question': torch.LongTensor([[data['question'] for _ in range(4)] for data in batch_data]),
                         'context': torch.LongTensor([data['context'] for data in batch_data]),
                         'type': [data['type'] for data in batch_data],
                         }
        if mode.startswith('train'):
            _collate_data['label'] = torch.LongTensor([data['label'] for data in batch_data])
        # 20211025开始使用参考书目文档
        if args.use_reference:
            _collate_data['subject'] = torch.LongTensor([data['subject'] for data in batch_data])
            _collate_data['reference'] = torch.LongTensor([[data['reference'] for _ in range(4)] for data in batch_data])
        return _collate_data

    dataloader = DataLoader(dataset=csvdataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=_collate_fn)
    return dataloader

# 训练数据集与测试数据集：处理生成的CSV文件
class CSVDataset(Dataset):

    def __init__(self, args, mode='train', export_data=False):
        # 构造变量转为成员变量
        self.args = args
        self.mode = mode
        self.export_data = export_data

        # 用于检索文档的语言模型变量
        self.base_demo()
        # self.base_demo_1()

    def base_demo(self):
        """最简单的处理，用有序编号值进行编码"""
        start_time = time.time()
        if self.mode.startswith('train'):
            filepaths = TRAINSET_PATHs[:]
        elif self.mode.startswith('test'):
            filepaths = TESTSET_PATHs[:]
        else:
            assert False, f'Unknown param `mode`: {self.mode}'
        max_option_length = self.args.max_option_length
        max_statement_length = self.args.max_statement_length
        max_reference_length = self.args.max_reference_length

        # 数据集预处理
        token2id_df = pandas.read_csv(TOKEN2ID_PATH, sep='\t', header=0)
        token2id = {token: _id for token, _id in zip(token2id_df['token'], token2id_df['id'])}
        dataset_dfs = [pandas.read_csv(filepath, sep='\t', header=0) for filepath in filepaths]
        dataset_df = pandas.concat(dataset_dfs).reset_index(drop=True)
        dataset_df['id'] = dataset_df['id'].astype(str)
        dataset_df['type'] = dataset_df['type'].astype(int)
        if self.mode.endswith('_kd'):
            dataset_df = dataset_df[dataset_df['type'] == 0].reset_index(drop=True)
        elif self.mode.endswith('_ca'):
            dataset_df = dataset_df[dataset_df['type'] == 1].reset_index(drop=True)
        else:
            dataset_df = dataset_df.reset_index(drop=True)

        dataset_df['question'] = dataset_df['statement'].map(self.token_to_id(max_statement_length, token2id))
        dataset_df['context'] = dataset_df[['option_a', 'option_b', 'option_c', 'option_d']].apply(self.combine_option(max_option_length, token2id), axis=1)

        print('处理题干和选项：', time.time() - start_time)

        if self.args.use_reference:
            self.model_sequence = load_language_model_sequence(model_names=self.args.language_model_names)
            self.dictionary = corpora.Dictionary.load(REFERENCE_DICTIONARY_PATH)
            self.similarity = generate_similarity(self.args, corpus_name=self.args.language_model_names[-1])
            self.stopwords = load_stopwords()
            # 使用参考书目文档将额外生成reference列
            reference_df = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0)
            self.index2subject = generate_index2subject(reference_df)

            print('读取参考书目生成index2subject：', time.time() - start_time)

            dataset_df['query_result'] = dataset_df[['statement', 'option_a', 'option_b', 'option_c', 'option_d']].apply(self.generate_query_result, axis=1)

            print('生成查询得分向量：', time.time() - start_time)

            dataset_df['reference_index'] = dataset_df['query_result'].map(lambda result: list(map(lambda x: x[0], result)))
            dataset_df['reference'] = dataset_df['reference_index'].map(self.preprocess_reference_index(max_reference_length, token2id, reference_df))

            print('确定参考段落：', time.time() - start_time)

            dataset_df['subject'] = dataset_df[['query_result', 'subject']].apply(self.fillna_subject, axis=1)  # 填充subject的缺失并预处理

            print('填充subject字段：', time.time() - start_time)

            if self.mode.startswith('train'):
                dataset_df['label'] = dataset_df['answer'].astype(int).tolist()
                self.data = dataset_df[['id', 'question', 'context', 'subject', 'reference', 'type', 'label']].reset_index(drop=True)
            elif self.mode.startswith('test'):
                self.data = dataset_df[['id', 'question', 'context', 'subject', 'reference', 'type']].reset_index(drop=True)
        else:
            if self.mode.startswith('train'):
                dataset_df['label'] = dataset_df['answer'].astype(int).tolist()
                self.data = dataset_df[['id', 'question', 'context', 'type', 'label']].reset_index(drop=True)
            elif self.mode.startswith('test'):
                self.data = dataset_df[['id', 'question', 'context', 'type']].reset_index(drop=True)

        if self.export_data:
            self.data.to_csv(COMPLETE_REFERENCE_PATH, sep='\t', header=True, index=False)

        print('导出数据完成：', time.time() - start_time)

    def base_demo_1(self):
        """直接读取存储好的CSV文件"""
        self.data = pandas.read_csv(COMPLETE_REFERENCE_PATH, sep='\t', header=0)
        self.data['question'] = self.data['question'].map(eval)
        self.data['context'] = self.data['context'].map(eval)
        self.data['subject'] = self.data['subject'].map(eval)
        self.data['reference'] = self.data['reference'].map(eval)
        self.data['label'] = self.data['label'].astype(int)

    def bert_demo(self):
        """使用BERT预训练模型进行语句编码"""
        pass

    def combine_option(self, max_length, token2id):
        """问题选项分词处理"""
        def _combine_option(_dataframe):
            def __token_to_id(__tokens):
                __ids = list(map(lambda __token: token2id.get(__token, token2id['UNK']), eval(__tokens)))
                if len(__ids) >= max_length:
                    return __ids[: max_length]
                else:
                    return __ids + [token2id['PAD']] * (max_length - len(__ids))

            return [__token_to_id(_dataframe[0]),
                    __token_to_id(_dataframe[1]),
                    __token_to_id(_dataframe[2]),
                    __token_to_id(_dataframe[3])]

        return _combine_option

    def token_to_id(self, max_length, token2id):
        """问题陈述分词处理"""
        def _token_to_id(_tokens):
            _ids = list(map(lambda _token: token2id.get(_token, token2id['UNK']), eval(_tokens)))
            if len(_ids) >= max_length:
                return _ids[: max_length]
            else:
                return _ids + [token2id['PAD']] * (max_length - len(_ids))

        return _token_to_id

    def preprocess_reference_index(self, max_length, token2id, reference_df):
        """预处理参考段落的索引"""
        def _preprocess_reference_index(_reference_index):
            _preprocessed_tokens_list = []
            for _index in _reference_index:
                _tokens = eval(reference_df.loc[_index, 'content'])
                _ids = list(map(lambda _token: token2id.get(_token, token2id['UNK']), _tokens))
                if len(_ids) >= max_length:
                    _preprocessed_tokens_list.append(_ids[: max_length])
                else:
                    _preprocessed_tokens_list.append(_ids + [token2id['PAD']] * (max_length - len(_ids)))
            return _preprocessed_tokens_list
        return _preprocess_reference_index

    def generate_query_result(self, series):
        """生成查询得分向量"""
        statement, option_a, option_b, option_c, option_d = list(map(lambda x: eval(x), series))
        query_tokens = statement + option_a + option_b + option_c + option_d
        if self.args.filter_stopword:
            query_tokens = filter_stopwords(query_tokens, stopwords=self.stopwords)
        result = query(query_tokens, self.dictionary, self.similarity, model_sequence=self.model_sequence)
        return result

    def fillna_subject(self, series):
        # 填充缺失的subject字段，这里拟填充为三个候选subject
        result, subject = series
        if subject == subject:
            return [SUBJECT2INDEX[subject]] + [0] * (self.args.top_subject - 1)
        if isinstance(result, str):
            result = eval(result)
        result2subject = [self.index2subject[entry[0]] for entry in result]
        weighted_count = {}
        for rank, subject in enumerate(result2subject):
            if subject in weighted_count:
                weighted_count[subject] += 1 / (rank + 1)
            else:
                weighted_count[subject] = 1 / (rank + 1)
        counter = Counter(weighted_count).most_common(self.args.top_subject)
        predicted_subjects = [x[0] for x in counter]
        return [SUBJECT2INDEX[subject] for subject in predicted_subjects] + [0] * (self.args.top_subject - len(predicted_subjects))

    def __getitem__(self, item):
        return self.data.loc[item, :]

    def __len__(self):
        return len(self.data)


