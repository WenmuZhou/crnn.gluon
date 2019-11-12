# -*- coding: utf-8 -*-
# @Time    : 2019/11/6 15:08
# @Author  : zhoujun
import numpy as np
from mxnet import image, nd
from mxnet.gluon.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, img_h: int, img_w: int, img_channel: int, num_label: int,
                 alphabet: str, ignore_chinese_punctuation, phase: str = 'train'):
        """
        数据集初始化
        :param data_txt: 存储着图片路径和对于label的文件
        :param data_shape: 图片的大小(h,w)
        :param img_channel: 图片通道数
        :param num_label: 最大字符个数,应该和网络最终输出的序列宽度一样
        :param alphabet: 字母表
        """
        super().__init__()
        assert phase in ['train', 'test']

        self.img_h = img_h
        self.img_w = img_w
        self.img_channel = img_channel
        self.num_label = num_label
        self.alphabet = alphabet
        self.phase = phase
        self.ignore_chinese_punctuation = ignore_chinese_punctuation
        self.label_dict = {}
        for i, char in enumerate(self.alphabet):
            self.label_dict[char] = i

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

    def pre_processing(self, img):
        """
        对图片进行处理，先按照高度进行resize，resize之后如果宽度不足指定宽度，就补黑色像素，否则就强行缩放到指定宽度
        :param img: 图片
        :return:
        """
        img_h = self.img_h
        img_w = self.img_w
        h, w = img.shape[:2]
        ratio_h = float(img_h) / h
        new_w = int(w * ratio_h)
        if new_w < img_w:
            img = image.imresize(img, w=new_w, h=img_h)
            step = nd.zeros((img_h, img_w - new_w, self.img_channel), dtype=img.dtype)
            img = nd.concat(img, step, dim=1)
        else:
            img = image.imresize(img, w=img_w, h=img_h)
        return img
