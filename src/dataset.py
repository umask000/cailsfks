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
import jieba
import torch
import pandas
import networkx

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


def token2frequency_to_csv(export_path, token2frequency):
    pandas.DataFrame({
        'token': list(token2frequency.keys()),
        'frequency': list(token2frequency.values()),
    }).sort_values(by=['frequency'], ascending=False).to_csv(export_path, sep='\t', index=False, header=True)


def reference_to_csv_1(export_path):
    reference_dict = {
        'law': [],
        'chapter_number_1': [],
        'chapter_number_2': [],
        'chapter_name_1': [],
        'chapter_name_2': [],
    }
    for i in range(TOTAL_REFERENCE_BLOCK):
        reference_dict[f'block_{i}'] = []

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
                        if line_string[-1] == '章':
                            chapter_name_2 = lines[i + 1].replace(' ', '')
                        else:
                            chapter_name_2 = line_string[line_string.find('章') + 1:]
                        break

                for i in range(total_lines):
                    blocks = lines[i].strip().split(' ')
                    total_blocks = len(blocks)
                    assert total_blocks <= TOTAL_REFERENCE_BLOCK
                    reference_dict['law'].append(law)
                    reference_dict['chapter_number_1'].append(chapter_number_1)
                    reference_dict['chapter_name_1'].append(chapter_name_1)
                    reference_dict['chapter_number_2'].append(chapter_number_2)
                    reference_dict['chapter_name_2'].append(chapter_name_2)
                    for j in range(TOTAL_REFERENCE_BLOCK):
                        reference_dict[f'block_{j}'].append(blocks[j] if j < total_blocks else '')

    reference_dataframe = pandas.DataFrame(reference_dict, columns=list(reference_dict.keys()))
    if export_path is not None:
        reference_dataframe.to_csv(export_path, sep='\t', header=True, index=False)
    return reference_dataframe


def reference_to_csv_2(export_path):
    reference_dict = {
        'law': [],
        'chapter_number': [],
        'chapter_name': [],
        'section': [],
        'content': [],
    }
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
                        if line_string[-1] == '章':
                            chapter_name_2 = lines[i + 1].replace(' ', '')
                        else:
                            chapter_name_2 = line_string[line_string.find('章') + 1:]
                        break
                chapter_number = chinese_to_number(chapter_number_1)
                chapter_name = chapter_name_1 if chapter_name_1 else chapter_name_2
                for i in range(total_lines):
                    blocks = lines[i].strip().split(' ')
                    section = ' '.join(blocks[: -1])
                    content = blocks[-1]
                    reference_dict['law'].append(law)
                    reference_dict['chapter_number'].append(chapter_number)
                    reference_dict['chapter_name'].append(chapter_name)
                    reference_dict['section'].append(section)
                    reference_dict['content'].append(content)

    reference_dataframe = pandas.DataFrame(reference_dict, columns=list(reference_dict.keys()))
    if export_path is not None:
        reference_dataframe.to_csv(export_path, sep='\t', header=True, index=False)
    return reference_dataframe


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


