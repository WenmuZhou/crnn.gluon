# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:20
# @Author  : zhoujun
import os
import argparse
import anyconfig
import numpy as np
from mxnet import nd
from mxnet.gluon.loss import CTCLoss

from models import get_model
from data_loader import get_dataloader
from trainer import Trainer
from utils import get_ctx


def main(config):
    if os.path.isfile(config['dataset']['alphabet']):
        config['dataset']['alphabet'] = str(np.load(config['dataset']['alphabet']))

    prediction_type = config['arch']['args']['prediction']['type']
    num_class = len(config['dataset']['alphabet'])

    # loss 设置
    if prediction_type == 'CTC':
        criterion = CTCLoss()
    else:
        raise NotImplementedError

    ctx = get_ctx(config['trainer']['gpus'])
    model = get_model(num_class, ctx, config['arch']['args'])
    model.hybridize()
    model.initialize(ctx=ctx)

    img_h, img_w = 32, 100
    for process in config['dataset']['train']['dataset']['args']['pre_processes']:
        if process['type'] == "Resize":
            img_h = process['args']['img_h']
            img_w = process['args']['img_w']
            break
    img_channel = 3 if config['dataset']['train']['dataset']['args']['img_mode'] != 'GRAY' else 1
    sample_input = nd.zeros((2, img_channel, img_h, img_w), ctx[0])
    num_label = model.get_batch_max_length(sample_input)

    train_loader = get_dataloader(config['dataset']['train'], num_label, config['dataset']['alphabet'])
    assert train_loader is not None
    if 'validate' in config['dataset']:
        validate_loader = get_dataloader(config['dataset']['validate'], num_label, config['dataset']['alphabet'])
    else:
        validate_loader = None

    config['lr_scheduler']['args']['step'] *= len(train_loader)

    trainer = Trainer(config=config,
                      model=model,
                      criterion=criterion,
                      train_loader=train_loader,
                      validate_loader=validate_loader,
                      sample_input = sample_input,
                      ctx=ctx)
    trainer.train()


def init_args():
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--config_file', default='config/icdar2015.yaml', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import sys

    project = 'crnn.gluon'  # 工作项目根目录
    sys.path.append(os.getcwd().split(project)[0] + project)

    from utils import parse_config

    args = init_args()
    assert os.path.exists(args.config_file)
    config = anyconfig.load(open(args.config_file, 'rb'))
    if 'base' in config:
        config = parse_config(config)
    main(config)
