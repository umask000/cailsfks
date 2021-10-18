# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn

import os
import time
import torch
import shutil
import logging

from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable

from setting import *
from config import TrainConfig

from src.model import BaseModel
from src.dataset import generate_dataloader
from src.utils import initialize_logging, load_args, save_args, save_checkpoint

if __name__ == '__main__':
    initialize_logging(filename=os.path.join(LOGGING_DIR, time.strftime('%Y%m%d')), filemode='w')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = load_args(TrainConfig)

    do_test = args.do_test
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    lr_multiplier = args.lr_multiplier
    weight_decay = args.weight_decay

    output_time = 1
    test_time = 1

    model = BaseModel(args).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_dataloader = generate_dataloader(args, mode='train')
    global_step = 0
    step_size = 1
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_multiplier)

    if do_test:
        test_dataloader = generate_dataloader(args, mode='test')

    total_len = len(train_dataloader)

    for epoch in range(n_epochs):
        logging.info(f'=== Epoch: {epoch} ===')
        total_loss = 0
        step = -1
        for step, data in enumerate(train_dataloader):
            for key in data.keys():
                if key == 'id':
                    continue
                data[key] = Variable(data[key]).to(device)
            optimizer.zero_grad()
            results = model(data, mode='train')
            loss, confusion_matrix = results['loss'], results['confusion matrix']
            total_loss += float(loss)
            loss.backward()
            optimizer.step()
            exp_lr_scheduler.step()
            logging.info(f'loss: {loss.item()} | confusion_matrix: {confusion_matrix}')
            global_step += 1
        if step == -1:
            logging.error('There is no data given to the model in this epoch, check your data.')
            raise NotImplementedError
        save_checkpoint(model=model, save_path=os.path.join(CHECKPOINT_DIR, f'{epoch}.h5'), optimizer=optimizer)