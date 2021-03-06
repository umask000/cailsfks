# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import sys
if __name__ == '__main__':
    import sys
    sys.path.append('../')

import torch
import logging
import argparse

from setting import *

# 初始化日志配置
def initialize_logging(filename, filemode='w'):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(filename)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=filename,
        filemode=filemode,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(filename)s | %(levelname)s | %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

# 加载停用词
def load_stopwords():
    stopwords = []
    for name, filepath in STOPWORD_PATHs.items():
        with open(filepath, 'r', encoding='utf8') as f:
            stopwords.extend(f.read().splitlines())
    return list(set(stopwords))

# 加载配置参数
def load_args(Config):
    config = Config()
    parser = config.parser
    try:
        return parser.parse_args()
    except:
        return parser.parse_known_args()[0]

# 保存配置参数
def save_args(args, save_path=None):

    class _MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, type) or isinstance(obj, types.FunctionType):
                return str(obj)
            return json.JSONEncoder.default(self, obj)

    if save_path is None:
        save_path = f'../logging/config_{time.strftime("%Y%m%d%H%M%S")}.json'
    with open(save_path, 'w') as f:
        f.write(json.dumps(vars(args), cls=_MyEncoder))

# 保存模型
def save_checkpoint(model,
                    save_path,
                    optimizer=None,
                    scheduler=None,
                    epoch=None,
                    iteration=None):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'iteration': iteration,
    }
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    torch.save(checkpoint, save_path)

# 加载模型
def load_checkpoint(model, save_path, optimizer=None, scheduler=None):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, checkpoint['epoch'], checkpoint['iteration']

# 编码问题答案：用1, 2, 4, 8分别表示四个选项并加和得到编码值，即二进制编码
def encode_answer(decoded_answer):
    return sum(list(map(lambda x: 2 ** OPTION2INDEX[x], filter(lambda x: x in OPTION2INDEX, decoded_answer))))

# 解码问题答案：encode_answer的反函数，即二进制解码
def decode_answer(encoded_answer):
    assert 16 > encoded_answer > 0
    decoded_answer = []
    for index, s in enumerate(bin(encoded_answer)[2:]):
        if s == '1':
            decoded_answer.append(INDEX2OPTION[index])
    return decoded_answer

# 中文数词转数字
def chinese_to_number(string):
    easy_mapper = {
        '一': '1',
        '二': '2',
        '两': '2',
        '三': '3',
        '四': '4',
        '五': '5',
        '六': '6',
        '七': '7',
        '八': '8',
        '九': '9',
        '十': '0',
    }
    number_string = ''.join(list(map(easy_mapper.get, string)))
    if number_string[0] == '0':
        number = int(number_string) + 10
    elif number_string[-1] == '0':
        number = int(number_string)
    else:
        number = int(number_string.replace('0', ''))
    return number

# 过滤停词
def filter_stopwords(tokens, stopwords=None):
    if stopwords is None:
        stopwords = load_stopwords()
    return list(filter(lambda x: not x in stopwords, tokens))

