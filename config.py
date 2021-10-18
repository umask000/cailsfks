# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import types
import argparse

from setting import *

class BaseConfig:
    parser = argparse.ArgumentParser("--")
    parser.add_argument('--num_blocks', default=10, type=int)


class TrainConfig(BaseConfig):
    BaseConfig.parser.add_argument('--num_workers', default=0, type=int)
    BaseConfig.parser.add_argument('--do_test', default=True, type=bool)
    BaseConfig.parser.add_argument('--local_rank', default=0, type=int)
    BaseConfig.parser.add_argument('--n_epochs', default=32, type=int)
    BaseConfig.parser.add_argument('--batch_size', default=4, type=int)
    BaseConfig.parser.add_argument('--frequency_threshold', default=FREQUENCY_THRESHOLD, type=int)
    BaseConfig.parser.add_argument('--max_statement_length', default=512, type=int)
    BaseConfig.parser.add_argument('--max_option_length', default=256, type=int)
    BaseConfig.parser.add_argument('--lr_multiplier', default=.95, type=float)
    BaseConfig.parser.add_argument('--learning_rate', default=.001, type=float)
    BaseConfig.parser.add_argument('--weight_decay', default=.0, type=float)


if __name__ == "__main__":
    config = TrainConfig()
    parser = config.parser
    args = parser.parse_args()
    print(args.frequency_threshold)
    print(args.logging)



