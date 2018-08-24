# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:18
# @Author  : zhoujun

import numpy as np
from mxnet import image, nd
import cv2
from mxnet.gluon.data import Dataset


class Gluon_OCRDataset(Dataset):
    def __init__(self, data_txt: str, data_shape: tuple, img_channel: int, num_label: int,
                 alphabet: str):
        """
        数据集初始化
        :param data_txt: 存储着图片路径和对于label的文件
        :param data_shape: 图片的大小(w,h)
        :param img_channel: 图片通道数
        :param num_label: 最大字符个数
        :param alphabet: 字母表
        """
        super(Gluon_OCRDataset, self).__init__()
        with open(data_txt, encoding='utf-8') as f:
            self.data_list = [x.replace('\n', '').split(' ') for x in f.readlines()]
        self.data_shape = data_shape
        self.img_channel = img_channel
        self.num_label = num_label
        self.alphabet = alphabet
        self.label_dict = {}
        for i, char in enumerate(self.alphabet):
            self.label_dict[char] = i

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        label = self.label_enocder(label)
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
        tmp_label = np.zeros(self.num_label, dtype=np.float32) - 1
        for i, ch in enumerate(label):
            tmp_label[i] = self.label_dict[ch]
        return tmp_label

    def pre_processing(self, img_path):
        """
        对图片进行处理，先按照高度进行resize，resize之后如果宽度不足指定宽度，就补黑色像素，否则就强行缩放到指定宽度
        :param img_path: 图片地址
        :return:
        """
        img = image.imdecode(open(img_path, 'rb').read(), 1 if self.img_channel == 3 else 0).astype(np.float32)
        h, w = img.shape[0], img.shape[1]
        size = (w, h)
        ratio_h = float(self.data_shape[1]) / size[1]
        new_size = tuple([int(x * ratio_h) for x in size])
        if new_size[0] < self.data_shape[0]:
            img = image.ForceResizeAug(new_size)(img)
            step = nd.zeros((self.data_shape[1], self.data_shape[0] - new_size[0], 3))
            img = nd.concat(img, step, dim=1)
        else:
            img = image.ForceResizeAug(self.data_shape)(img)
        return img


if __name__ == '__main__':
    import keys
    from mxnet.gluon.data import DataLoader
    from matplotlib import pyplot as plt
    from mxnet.gluon.data.vision.transforms import ToTensor

    alphabet = keys.alphabet
    dataset = Gluon_OCRDataset(r'Z:\lsp\lsp\number_crnn\crnn\data\test_win.txt', (320, 32), 3, 81, alphabet)
    data_loader = DataLoader(dataset.transform_first(ToTensor()), 1,shuffle=True)
    for img, label in data_loader:
        label = label[0].asnumpy()
        label = ''.join([alphabet[int(i)] for i in label[label != -1]])
        img = img[0].asnumpy().transpose(1, 2, 0)
        plt.title(label)
        plt.imshow(img)
        plt.show()
        break
