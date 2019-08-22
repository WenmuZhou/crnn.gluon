# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:18
# @Author  : zhoujun
import numpy as np
from mxnet import image, nd, recordio
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data import Dataset, RecordFileDataset


class ImageDataset(Dataset):
    def __init__(self, data_list: list, img_h: int, img_w: int, img_channel: int, num_label: int,
                 alphabet: str, phase: str = 'train'):
        """
        数据集初始化
        :param data_txt: 存储着图片路径和对于label的文件
        :param data_shape: 图片的大小(h,w)
        :param img_channel: 图片通道数
        :param num_label: 最大字符个数,应该和网络最终输出的序列宽度一样
        :param alphabet: 字母表
        """
        super(ImageDataset, self).__init__()
        assert phase in ['train', 'test']

        self.data_list = data_list
        self.img_h = img_h
        self.img_w = img_w
        self.img_channel = img_channel
        self.num_label = num_label
        self.alphabet = alphabet
        self.phase = phase
        self.label_dict = {}
        for i, char in enumerate(self.alphabet):
            self.label_dict[char] = i

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        label = label.replace(' ', '')
        try:
            label = self.label_enocder(label)
        except Exception as e:
            print(img_path, label)
        img = self.pre_processing(img_path)
        return img, label

    def __len__(self):
        return len(self.data_list)

    def label_enocder(self, label):
        """
        对label进行处理，将输入的label字符串转换成在字母表中的索引
        :param label: label字符串
        :return: 索引列表
        """
        tmp_label = nd.zeros(self.num_label, dtype=np.float32) - 1
        for i, ch in enumerate(label):
            tmp_label[i] = self.label_dict[ch]
        return tmp_label

    def pre_processing(self, img_path):
        """
        对图片进行处理，先按照高度进行resize，resize之后如果宽度不足指定宽度，就补黑色像素，否则就强行缩放到指定宽度

        :param img_path: 图片地址
        :return:
        """
        data_augment = False
        if self.phase == 'train' and np.random.rand() > 0.5:
            data_augment = True
        if data_augment:
            img_h = 40
            img_w = 340
        else:
            img_h = self.img_h
            img_w = self.img_w
        img = image.imdecode(open(img_path, 'rb').read(), 1 if self.img_channel == 3 else 0)
        h, w = img.shape[:2]
        ratio_h = float(img_h) / h
        new_w = int(w * ratio_h)
        if new_w < img_w:
            img = image.imresize(img, w=new_w, h=img_h)
            step = nd.zeros((img_h, img_w - new_w, self.img_channel), dtype=img.dtype)
            img = nd.concat(img, step, dim=1)
        else:
            img = image.imresize(img, w=img_w, h=img_h)

        if data_augment:
            img, _ = image.random_crop(img, (self.img_w, self.img_h))
        return img


class RecordDataset(RecordFileDataset):
    """
    A dataset wrapping over a RecordIO file contraining images
    Each sample is an image and its corresponding label
    """

    def __init__(self, filename, data_shape: tuple, img_channel: int, num_label: int):
        super(RecordDataset, self).__init__(filename)
        self.data_shape = data_shape
        self.img_channel = img_channel
        self.num_label = num_label

    def __getitem__(self, idx):
        record = super(RecordDataset, self).__getitem__(idx)
        header, img = recordio.unpack(record)
        img = self.pre_processing(img)
        label = self.label_enocder(header.label)
        return img, label

    def label_enocder(self, label):
        """
        对label进行处理，将输入的label字符串转换成在字母表中的索引
        :param label: label字符串
        :return: 索引列表
        """
        label = nd.array(label)
        tmp_label = nd.zeros(self.num_label - len(label), dtype=np.float32) - 1
        label = nd.concat(label, tmp_label, dim=0)
        return label

    def pre_processing(self, img):
        """
        对图片进行处理
        :param img_path: 图片
        :return:
        """
        img = image.imdecode(img, 1 if self.img_channel == 3 else 0)
        h, w = img.shape[:2]
        ratio_h = float(self.data_shape[0]) / h
        new_w = int(w * ratio_h)
        if new_w < self.data_shape[1]:
            img = image.imresize(img, w=new_w, h=self.data_shape[0])
            step = nd.zeros((self.data_shape[0], self.data_shape[1] - new_w, self.img_channel), dtype=img.dtype)
            img = nd.concat(img, step, dim=1)
        else:
            img = image.imresize(img, w=self.data_shape[1], h=self.data_shape[0])
        return img


class Batch_Balanced_Dataset(object):
    def __init__(self, dataset_list: list, ratio_list: list, module_args: dict, dataset_transfroms,
                 phase: str = 'train'):
        """
        对datasetlist里的dataset按照ratio_list里对应的比例组合，似的每个batch里的数据按按照比例采样的
        :param dataset_list: 数据集列表
        :param ratio_list: 比例列表
        :param module_args: dataloader的配置
        :param dataset_transfroms: 数据集使用的transforms
        :param phase: 训练集还是验证集
        """
        assert sum(ratio_list) == 1 and len(dataset_list) == len(ratio_list)

        self.dataset_len = 0
        self.data_loader_list = []
        self.dataloader_iter_list = []
        all_batch_size = module_args['loader']['train_batch_size'] if phase == 'train' else module_args['loader'][
            'val_batch_size']
        for _dataset, batch_ratio_d in zip(dataset_list, ratio_list):
            _batch_size = max(round(all_batch_size * float(batch_ratio_d)), 1)

            _data_loader = DataLoader(dataset=_dataset.transform_first(dataset_transfroms),
                                      batch_size=_batch_size,
                                      shuffle=module_args['loader']['shuffle'],
                                      last_batch='rollover',
                                      num_workers=module_args['loader']['num_workers'])
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))
            self.dataset_len += len(_dataset)

    def __iter__(self):
        return self

    def __len__(self):
        return min([len(x) for x in self.data_loader_list])

    def __next__(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = next(data_loader_iter)
                balanced_batch_images.append(image)
                balanced_batch_texts.append(text)
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = next(self.dataloader_iter_list[i])
                balanced_batch_images.append(image)
                balanced_batch_texts.append(text)
            except ValueError:
                pass

        balanced_batch_images = nd.concat(*balanced_batch_images, dim=0)
        balanced_batch_texts = nd.concat(*balanced_batch_texts, dim=0)
        return balanced_batch_images, balanced_batch_texts
