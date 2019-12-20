# -*- coding: utf-8 -*-
# @Time    : 18-11-16 下午5:46
# @Author  : zhoujun
import copy
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms

def get_dataset(data_path, module_name, dataset_args):
    """
    获取训练dataset
    :param data_path: dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :param module_name: 所使用的自定义dataset名称，目前只支持data_loaders.ImageDataset
    :param transform: 该数据集使用的transforms
    :param dataset_args: module_name的参数
    :return: 如果data_path列表不为空，返回对应的Dataset对象，否则None
    """
    from . import dataset
    s_dataset = getattr(dataset, module_name)(data_path=data_path, **dataset_args)
    return s_dataset


def get_transforms(transforms_config):
    tr_list = []
    for item in transforms_config:
        if 'args' not in item:
            args = {}
        else:
            args = item['args']
        cls = getattr(transforms, item['type'])(**args)
        tr_list.append(cls)
    tr_list = transforms.Compose(tr_list)
    return tr_list


def get_dataloader(module_config, num_label, alphabet):
    if module_config is None:
        return None
    config = copy.deepcopy(module_config)
    dataset_args = config['dataset']['args']
    dataset_args['num_label'] = num_label
    dataset_args['alphabet'] = alphabet
    if 'transforms' in dataset_args:
        img_transfroms = get_transforms(dataset_args.pop('transforms'))
    else:
        img_transfroms = None
    # 创建数据集
    dataset_name = config['dataset']['type']
    data_path_list = dataset_args.pop('data_path')
    if 'data_ratio' in dataset_args:
        data_ratio = dataset_args.pop('data_ratio')
    else:
        data_ratio = [1.0]

    _dataset_list = []
    for data_path in data_path_list:
        _dataset_list.append(get_dataset(data_path=data_path, module_name=dataset_name, dataset_args=dataset_args))
    if len(data_ratio) > 1 and len(dataset_args['data_ratio']) == len(_dataset_list):
        from . import dataset
        loader = dataset.Batch_Balanced_Dataset(dataset_list=_dataset_list, ratio_list=data_ratio, loader_args=config['loader'],
                                                dataset_transfroms=img_transfroms, phase='train')
    else:
        _dataset = _dataset_list[0]
        loader = DataLoader(dataset=_dataset.transform_first(img_transfroms), **config['loader'])
        loader.dataset_len = len(_dataset)
    return loader
