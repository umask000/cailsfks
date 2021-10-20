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

from collections import Counter
from torch.utils.data import Dataset, DataLoader

from setting import *
from config import TrainConfig
from src.utils import load_args, encode_answer, chinese_to_number

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

# 参考文献TXT文本转为CSV格式文件
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

# 给参考文献制作文档索引：正排索引和倒排索引
def index_reference():
    reference_dataframe = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0)
    reference_token2id = pandas.read_csv(REFERENCE_TOKEN2ID_PATH, sep='\t', header=0)

    document2id = {}    # 字典键为文档四元组(law, chapter_number, chapter_name, section), 值为文档编号
    id2document = {}    # 字典键为文档编号, 值为文档四元组(law, chapter_number, chapter_name, section)

    forward_index = {}                                                      # 前向索引字典: 键为文档编号, 值为四元组(分词词频信息, 分词段落集合, 文档最高频词的词频, 文档总词数)
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

def generate_dataloader(args, mode='train'):
    assert mode in ['train', 'test'], f'Unknown param `mode`: {mode}'
    batch_size = args.batch_size
    num_workers = args.num_workers
    csvdataset = CSVDataset(args, mode=mode)

    def _collate_fn(batch_data):
        _collate_data = {'id': [data['id'] for data in batch_data],
                         'question': torch.LongTensor([[data['question'] for _ in range(4)] for data in batch_data]),
                         'context': torch.LongTensor([data['context'] for data in batch_data])}
        if mode == 'train':
            _collate_data['label'] = torch.LongTensor([data['label'] for data in batch_data])
        return _collate_data

    dataloader = DataLoader(dataset=csvdataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=_collate_fn)
    return dataloader


class CSVDataset(Dataset):

    def __init__(self, args, mode='train'):
        if mode == 'train':
            filepaths = TRAINSET_PATHs[:]
        elif mode == 'test':
            filepaths = TESTSET_PATHs[:]
        else:
            assert False, f'Unknown param `mode`: {mode}'
        max_option_length = args.max_option_length
        max_statement_length = args.max_statement_length
        dataset_dfs = [pandas.read_csv(filepath, sep='\t', header=0) for filepath in filepaths]
        dataset_df = pandas.concat(dataset_dfs)
        token2id_df = pandas.read_csv(TOKEN2ID_PATH, sep='\t', header=0)
        token2id = {token: _id for token, _id in zip(token2id_df['token'], token2id_df['id'])}

        dataset_df['id'] = dataset_df['id'].astype(str)
        dataset_df['question'] = dataset_df['statement'].map(self.token_to_id(max_statement_length, token2id))
        dataset_df['context'] = dataset_df[['option_a', 'option_b', 'option_c', 'option_d']].apply(
            self.combine_option(max_option_length, token2id), axis=1)
        if mode == 'train':
            dataset_df['label'] = dataset_df['answer'].astype(int).tolist()
            self.data = dataset_df[['id', 'question', 'context', 'label']].reset_index(drop=True)
        elif mode == 'test':
            self.data = dataset_df[['id', 'question', 'context']].reset_index(drop=True)

    def combine_option(self, max_length, token2id):
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
        def _token_to_id(_tokens):
            _ids = list(map(lambda _token: token2id.get(_token, token2id['UNK']), eval(_tokens)))
            if len(_ids) >= max_length:
                return _ids[: max_length]
            else:
                return _ids + [token2id['PAD']] * (max_length - len(_ids))

        return _token_to_id

    def __getitem__(self, item):
        return self.data.loc[item, :]

    def __len__(self):
        return len(self.data)


