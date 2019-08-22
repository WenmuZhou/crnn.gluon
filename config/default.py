# -*- coding: utf-8 -*-
# @Time    : 18-11-16 上午11:03
# @Author  : zhoujun

name = 'crnn'
arch = {
    "type": "crnnModel",  # name of model architecture to train
    "args": {
        'feature_extraction': {
            'type': 'VGG',  # VGG ,RCNN or ResNet
        },
        'sequence_model': {
            'type': 'RNN',  # RNN
            'args': {
                'hidden_size': 512,
                'num_layers': 1
            }
        },
        'prediction': {
            'type': 'CTC',  # CTC or Attn
            'args': {
            }
        }
    }
}

data_loader = {
    "type": "ImageDataset",  # selecting data loader
    "args": {
        'dataset': {
            'train_data_path': [['dataset1.txt1', 'dataset1.txt2'], ['dataset2.txt1', 'dataset2.txt2']],
            'train_data_ratio': [0.5, 0.5],
            'val_data_path': ['val.txt'],
            'img_h': 32,
            'img_w': 320,
            'img_channel': 3,
            'num_label': 80,
            'alphabet': 'alphabet.npy',
        },
        'loader': {
            'validation_split': 0.1,
            'train_batch_size': 16,
            'val_batch_size': 4,
            'shuffle': True,
            'num_workers': 6
        }
    }
}

optimizer = {
    "type": "Adam",
    "args": {
        "learning_rate": 0.001,
    }
}

lr_scheduler = {
    "type": "FactorScheduler",
    "args": {
        "step": 30,
        "factor": 0.1,
        'stop_factor_lr': 1e-7,
        'warmup_begin_lr': 1e-4
    }
}

resume = {
    'restart_training': True,
    'checkpoint': ''
}

trainer = {
    # random seed
    'seed': 2,
    'gpus': 0,
    'epochs': 100,
    'display_interval': 10,
    'resume': resume,
    'output_dir': 'output',
    'tensorboard': True
}

config_dict = {}
config_dict['name'] = name
config_dict['data_loader'] = data_loader
config_dict['arch'] = arch
config_dict['optimizer'] = optimizer
config_dict['lr_scheduler'] = lr_scheduler
config_dict['trainer'] = trainer

from utils import save_json

save_json(config_dict, '../config.json')
