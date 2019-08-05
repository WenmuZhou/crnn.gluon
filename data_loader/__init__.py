# -*- coding: utf-8 -*-
# @Time    : 18-11-16 下午5:46
# @Author  : zhoujun
import pathlib
import copy
import random
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms

from . import data_loaders


def get_dataset(data_list, module_name, phase, dataset_args):
    """
    获取训练dataset
    :param data_path: dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :param module_name: 所使用的自定义dataset名称，目前只支持data_loaders.ImageDataset
    :param phase: 是train or test 阶段
    :param dataset_args: module_name的参数
    :return: 如果data_path列表不为空，返回对于的ConcatDataset对象，否则None
    """
    dataset = getattr(data_loaders, module_name)(phase=phase, data_list=data_list,
                                                 **dataset_args)
    return dataset


def get_datalist(train_data_path, val_data_path, validation_split=0.1):
    """
    获取训练和验证的数据list
    :param train_data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :param val_data_path: 验证的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :param validation_split: 验证集的比例，当val_data_path为空时使用
    :return:
    """
    train_data_list = []
    for p in train_data_path:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                if len(line) > 1:
                    img_path = pathlib.Path(line[0])
                    if img_path.exists() and img_path.stat().st_size > 0 and line[1]:
                        train_data_list.append((line[0], line[1]))

    val_data_list = []
    for p in val_data_path:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                if len(line) > 1:
                    img_path = pathlib.Path(line[0])
                    if img_path.exists() and img_path.stat().st_size > 0 and line[1]:
                        val_data_list.append((line[0], line[1]))

    if len(val_data_path) == 0:
        val_len = int(len(train_data_list) * validation_split)
        random.shuffle(train_data_list)
        val_data_list = train_data_list[:val_len]
        train_data_list = train_data_list[val_len:]
    return train_data_list, val_data_list


def get_dataloader(module_name, module_args, num_label):
    train_transfroms = transforms.Compose([
        transforms.RandomColorJitter(brightness=0.5),
        transforms.ToTensor()
    ])

    val_transfroms = transforms.ToTensor()
    # 创建数据集
    dataset_args = module_args['dataset']
    train_data_path = dataset_args.pop('train_data_path')
    val_data_path = dataset_args.pop('val_data_path')

    train_data_list, val_data_list = get_datalist(train_data_path, val_data_path,
                                                  module_args['loader']['validation_split'])

    dataset_args['num_label'] = num_label
    train_dataset = get_dataset(data_list=train_data_list,
                                module_name=module_name,
                                phase='train',
                                dataset_args=dataset_args)

    val_dataset = get_dataset(data_list=val_data_list,
                              module_name=module_name,
                              phase='test',
                              dataset_args=dataset_args)

    train_loader = DataLoader(dataset=train_dataset.transform_first(train_transfroms),
                              batch_size=module_args['loader']['train_batch_size'],
                              shuffle=module_args['loader']['shuffle'],
                              last_batch='keep')
 #                             num_workers=module_args['loader']['num_workers'])

    val_loader = DataLoader(dataset=val_dataset.transform_first(train_transfroms),
                            batch_size=module_args['loader']['train_batch_size'],
                            shuffle=module_args['loader']['shuffle'],
                            last_batch='keep')
#                            num_workers=module_args['loader']['num_workers'])

    return train_loader, val_loader
