# -*- coding: utf-8 -*-
# @Time    : 18-11-16 下午5:46
# @Author  : zhoujun
import pathlib
import random
from tqdm import tqdm
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms

from . import dataset


def get_datalist(train_data_path, val_data_path, validation_split=0.1):
    """
    获取训练和验证的数据list
    :param train_data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :param val_data_path: 验证的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :param validation_split: 验证集的比例，当val_data_path为空时使用
    :return:
    """
    train_data_list = []
    for train_path in train_data_path:
        train_data = []
        for p in train_path:
            with open(p, 'r', encoding='utf-8') as f:
                for line in tqdm(f.readlines(), desc='reading data:' + p):
                    line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                    if len(line) > 1:
                        img_path = pathlib.Path(line[0])
                        if img_path.exists() and img_path.stat().st_size > 0 and line[1]:
                            train_data.append((line[0], line[1]))
        train_data_list.append(train_data)

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
        for i, train_data in enumerate(train_data_list):
            val_len = int(len(train_data) * validation_split)
            random.shuffle(train_data)
            val_data_list.extend(train_data[:val_len])
            train_data_list[i] = train_data[val_len:]
    return train_data_list, val_data_list


def get_dataset(data_list, module_name, phase, dataset_args):
    """
    获取训练dataset
    :param data_path: dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :param module_name: 所使用的自定义dataset名称，目前只支持data_loaders.ImageDataset
    :param phase: 是train or test 阶段
    :param dataset_args: module_name的参数
    :return: 如果data_path列表不为空，返回对于的ConcatDataset对象，否则None
    """
    s_dataset = getattr(dataset, module_name)(phase=phase, data_list=data_list,
                                              **dataset_args)
    return s_dataset


def get_dataloader(module_name, module_args, num_label):
    train_transfroms = transforms.Compose([
        transforms.RandomColorJitter(brightness=0.5),
        transforms.ToTensor()
    ])

    val_transfroms = transforms.ToTensor()
    dataset_args = module_args['dataset']
    dataset_args['num_label'] = num_label
    # 创建数据集
    train_data_path = dataset_args.pop('train_data_path')
    train_data_ratio = dataset_args.pop('train_data_ratio')
    val_data_path = dataset_args.pop('val_data_path')

    if module_name == 'ImageDataset':
        train_data_list, val_data_list = get_datalist(train_data_path, val_data_path,
                                                      module_args['loader']['validation_split'])
    elif module_name == 'LmdbDataset':
        train_data_list = train_data_path
        val_data_list = val_data_path
    else:
        raise Exception('current only support ImageDataset and LmdbDataset')
    train_dataset_list = []
    for train_data in train_data_list:
        train_dataset_list.append(get_dataset(data_list=train_data,
                                              module_name=module_name,
                                              phase='train',
                                              dataset_args=dataset_args))

    if len(train_dataset_list) > 1:
        train_loader = dataset.Batch_Balanced_Dataset(dataset_list=train_dataset_list,
                                                      ratio_list=train_data_ratio,
                                                      module_args=module_args,
                                                      dataset_transfroms=train_transfroms,
                                                      phase='train')
    elif len(train_dataset_list) == 1:
        train_loader = DataLoader(dataset=train_dataset_list[0].transform_first(train_transfroms),
                                  batch_size=module_args['loader']['train_batch_size'],
                                  shuffle=module_args['loader']['shuffle'],
                                  last_batch='rollover',
                                  num_workers=module_args['loader']['num_workers'])
        train_loader.dataset_len = len(train_dataset_list[0])
    else:
        raise Exception('no images found')
    if len(val_data_list):
        val_dataset = get_dataset(data_list=val_data_list,
                                  module_name=module_name,
                                  phase='test',
                                  dataset_args=dataset_args)
        val_loader = DataLoader(dataset=val_dataset.transform_first(val_transfroms),
                                batch_size=module_args['loader']['val_batch_size'],
                                shuffle=module_args['loader']['shuffle'],
                                last_batch='keep',
                                num_workers=module_args['loader']['num_workers'])
        val_loader.dataset_len = len(val_dataset)
    else:
        val_loader = None
    return train_loader, val_loader
