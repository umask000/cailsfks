# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import sys
if __name__ == '__main__':
    import sys
    sys.path.append('../')

import torch
import pandas

from torch import nn
from torch.nn import Module, Embedding, Linear, CrossEntropyLoss, functional as F

from setting import *
from src.utils import decode_answer
from src.module import BaseLSTMEncoder, BaseAttention

class BaseModel(Module):

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.d_hidden = args.max_option_length
        self.n_tokens = pandas.read_csv(TOKEN2ID_PATH, sep='\t', header=0).shape[0]
        self.confusion_matrix = []
        self.embedding = Embedding(self.n_tokens, self.d_hidden)
        self.context_encoder = BaseLSTMEncoder(args)
        self.question_encoder = BaseLSTMEncoder(args)
        self.attention = BaseAttention(args)
        self.rank_module = Linear(self.d_hidden * 2, 1)
        self.criterion = CrossEntropyLoss()
        self.multi_module = Linear(4, 16)

    def forward(self, data, mode='train'):
        assert mode in ['train', 'test'], f'Unknown param `mode`: {mode}'
        context = data['context']
        question = data['question']
        batch_size = question.size()[0]
        n_options = question.size()[1]
        embedded_context = self.embedding(context.view(batch_size * n_options, -1))
        embedded_question = self.embedding(question.view(batch_size * n_options, -1))
        _, encoded_context = self.context_encoder(embedded_context)
        _, encoded_question = self.question_encoder(embedded_question)
        context_attention, question_attention, attention = self.attention(encoded_context, encoded_question)
        y = torch.cat([torch.max(context_attention, dim=1)[0], torch.max(question_attention, dim=1)[0]], dim=1)
        y = self.rank_module(y.view(batch_size * n_options, -1))
        output = self.multi_module(y.view(batch_size, n_options))
        if mode == 'test':
            result = []
            for _id, _output in zip(data['id'], output):
                encoded_answer = int(torch.max(_output, dim=0)[1])
                answer = [INDEX2OPTION[encoded_answer]] if len(_output) == 4 else decode_answer(encoded_answer)
                result.append({'id': _id, 'answer': answer})
            return result
        elif mode == 'train':
            label = data['label']
            loss = self.criterion(output, label)
            self.single_label_top1_accuracy(output, label)
            return {'loss': loss, 'confusion matrix': self.confusion_matrix}

    def single_label_top1_accuracy(self, output, label):
        encoded_answers = torch.max(output, dim=1)[1]
        n_classes = output.size(1)
        while len(self.confusion_matrix) < n_classes:
            self.confusion_matrix.append({'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0})
        for encoded_answer, _label in zip(encoded_answers, label):
            y_pred = int(encoded_answer)
            y_true = int(_label)
            if y_pred == y_true:
                self.confusion_matrix[y_pred]['TP'] += 1
            else:
                self.confusion_matrix[y_pred]['FP'] += 1
                self.confusion_matrix[y_true]['FN'] += 1


class RCModel(Module):

    def __init__(self, args):
        super(RCModel, self).__init__()
        self.d_hidden = args.max_option_length
        self.n_tokens = pandas.read_csv(TOKEN2ID_PATH, sep='\t', header=0).shape[0]
        self.confusion_matrix = []
        self.embedding = Embedding(self.n_tokens, self.d_hidden)
        self.context_encoder = BaseLSTMEncoder(args)
        self.question_encoder = BaseLSTMEncoder(args)

        self.reference_encoder = BaseLSTMEncoder(args)
        self.subject_encoder = BaseLSTMEncoder(args)

        self.attention = BaseAttention(args)
        self.rank_module = Linear(self.d_hidden * 6, 1)
        self.criterion = CrossEntropyLoss()
        self.multi_module = Linear(4, 16)

    def forward(self, data, mode='train'):
        assert mode in ['train', 'test'], f'Unknown param `mode`: {mode}'
        context = data['context']
        question = data['question']
        reference = data['reference']
        subject = data['subject']
        batch_size = question.size()[0]
        n_options = question.size()[1]
        embedded_context = self.embedding(context.view(batch_size * n_options, -1))
        embedded_question = self.embedding(question.view(batch_size * n_options, -1))
        embedded_reference = self.embedding(reference.view(batch_size * n_options, -1))

        _, encoded_context = self.context_encoder(embedded_context)
        _, encoded_question = self.question_encoder(embedded_question)
        _, encoded_reference = self.reference_encoder(embedded_reference)

        context_attention_1, question_attention_1, _ = self.attention(encoded_context, encoded_question)
        context_attention_2, reference_attention_1, _ = self.attention(encoded_context, encoded_reference)
        question_attention_2, reference_attention_2, _ = self.attention(encoded_question, encoded_reference)

        y1 = torch.cat([torch.max(context_attention_1, dim=1)[0], torch.max(question_attention_1, dim=1)[0]], dim=1)
        y2 = torch.cat([torch.max(context_attention_2, dim=1)[0], torch.max(reference_attention_1, dim=1)[0]], dim=1)
        y3 = torch.cat([torch.max(question_attention_2, dim=1)[0], torch.max(reference_attention_2, dim=1)[0]], dim=1)
        y = torch.cat([y1, y2, y3], dim=1)
        y = self.rank_module(y.view(batch_size * n_options, -1))
        output = self.multi_module(y.view(batch_size, n_options))
        if mode == 'test':
            result = []
            for _id, _output in zip(data['id'], output):
                encoded_answer = int(torch.max(_output, dim=0)[1])
                answer = [INDEX2OPTION[encoded_answer]] if len(_output) == 4 else decode_answer(encoded_answer)
                result.append({'id': _id, 'answer': answer})
            return result
        elif mode == 'train':
            label = data['label']
            loss = self.criterion(output, label)
            self.single_label_top1_accuracy(output, label)
            return {'loss': loss, 'confusion matrix': self.confusion_matrix}

    def single_label_top1_accuracy(self, output, label):
        encoded_answers = torch.max(output, dim=1)[1]
        n_classes = output.size(1)
        while len(self.confusion_matrix) < n_classes:
            self.confusion_matrix.append({'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0})
        for encoded_answer, _label in zip(encoded_answers, label):
            y_pred = int(encoded_answer)
            y_true = int(_label)
            if y_pred == y_true:
                self.confusion_matrix[y_pred]['TP'] += 1
            else:
                self.confusion_matrix[y_pred]['FP'] += 1
                self.confusion_matrix[y_true]['FN'] += 1