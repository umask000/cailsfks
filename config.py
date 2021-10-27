# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import types
import argparse

from setting import *

class BaseConfig:
    parser = argparse.ArgumentParser("--")


class TrainConfig(BaseConfig):
    BaseConfig.parser.add_argument('--num_workers', default=0, type=int)
    BaseConfig.parser.add_argument('--do_test', default=True, type=bool)
    BaseConfig.parser.add_argument('--n_epochs', default=32, type=int)
    BaseConfig.parser.add_argument('--batch_size', default=4, type=int)
    BaseConfig.parser.add_argument('--frequency_threshold', default=FREQUENCY_THRESHOLD, type=int)
    BaseConfig.parser.add_argument('--max_reference_length', default=256, type=int, help='经检测，512超过参考段落分词长度的0.998分位数')
    BaseConfig.parser.add_argument('--max_statement_length', default=256, type=int)
    BaseConfig.parser.add_argument('--max_option_length', default=128, type=int)
    BaseConfig.parser.add_argument('--lr_multiplier', default=.95, type=float)
    BaseConfig.parser.add_argument('--learning_rate', default=.001, type=float)
    BaseConfig.parser.add_argument('--weight_decay', default=.0, type=float)

    BaseConfig.parser.add_argument('--use_reference', default=False, type=bool, help='是否使用参考书目（20211025以后默认使用）')
    BaseConfig.parser.add_argument('--language_model_names', default=['tfidf'], type=list, help='使用的语言模型序列，如果使用LSI模型，同样需要先调用TFIDF')

    BaseConfig.parser.add_argument('--filter_stopword', default=True, type=int, help='是否过滤停用词')

    BaseConfig.parser.add_argument('--lsi_num_topics', default=64, type=int)
    BaseConfig.parser.add_argument('--lsi_num_best', default=16, type=int)
    BaseConfig.parser.add_argument('--lda_num_topics', default=64, type=int)
    BaseConfig.parser.add_argument('--lda_num_best', default=16, type=int)

    BaseConfig.parser.add_argument('--top_subject', default=3, type=int, help='预测的subject数量（只预测一个的正确率太低了，一般预测三个候选）')

if __name__ == "__main__":
    config = BigConfig()
    parser = config.parser
    args = parser.parse_args()
    print(args.num_workers)



