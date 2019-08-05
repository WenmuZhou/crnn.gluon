# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:20
# @Author  : zhoujun
import os
import numpy as np
from mxnet.gluon.loss import CTCLoss

from models import get_model
from data_loader import get_dataloader
from trainer import Trainer
from utils import read_json, try_gpu


def main(config):
    if os.path.isfile(config['data_loader']['args']['dataset']['alphabet']):
        config['data_loader']['args']['dataset']['alphabet'] = str(np.load('alphabet.npy'))
    config['data_loader']['args']['dataset']['alphabet'] += '嫑'

    prediction_type = config['arch']['args']['prediction']['type']
    num_class = len(config['data_loader']['args']['dataset']['alphabet'])

    # loss 设置
    if prediction_type == 'CTC':
        criterion = CTCLoss()
    else:
        raise NotImplementedError

    ctx = try_gpu(config['trainer']['gpus'])
    model = get_model(num_class, config['arch']['args'])
    model.hybridize()
    model.initialize(ctx=ctx)

    img_w = config['data_loader']['args']['dataset']['img_w']
    img_h = config['data_loader']['args']['dataset']['img_h']
    train_loader, val_loader = get_dataloader(config['data_loader']['type'], config['data_loader']['args'],
                                               num_label=model.get_batch_max_length(img_h=img_h, img_w=img_w, ctx=ctx))

    config['lr_scheduler']['args']['step'] *= len(train_loader)
    config['name'] = config['name'] + '_' + model.model_name
    trainer = Trainer(config=config,
                      model=model,
                      criterion=criterion,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      ctx=ctx)
    trainer.train()


if __name__ == '__main__':
    config = read_json('config.json')
    main(config)
