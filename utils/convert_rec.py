# -*- coding: utf-8 -*-
# @Time    : 18-8-31 上午9:48
# @Author  : zhoujun

import os
import pathlib
from tqdm import tqdm
import numpy as np
import mxnet as mx
import cv2


def get_datalist(train_data_path):
    """
    获取训练和验证的数据list
    :param train_data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
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
    return train_data_list


def pre_processing(img_path, data_shape, img_channel):
    """
    对图片进行处理，先按照高度进行resize，resize之后如果宽度不足指定宽度，就补黑色像素，否则就强行缩放到指定宽度
    :param img_channel:
    :param data_shape:
    :param img_path: 图片地址
    :return:
    """
    img = cv2.imdecode(np.fromfile(img_path), 1)
    img = cv2.resize(img, (110, 16))
    h, w = img.shape[:2]
    ratio_h = float(data_shape[1]) / h
    new_w = int(w * ratio_h)
    if new_w < data_shape[0]:
        img = cv2.resize(img, (new_w, data_shape[1]))
        step = np.zeros((data_shape[1], data_shape[0] - new_w, img_channel), dtype=img.dtype)
        img = np.column_stack((img, step))
    else:
        img = cv2.resize(img, tuple(data_shape))
    return img

def label_enocder(label,num_label):
    """
    对label进行处理，将输入的label字符串转换成在字母表中的索引
    :param label: label字符串
    :return: 索引列表
    """
    tmp_label = np.zeros(num_label, dtype=np.float32) - 1
    for i, ch in enumerate(label):
        tmp_label[i] = label_dict[ch]
    return tmp_label

def make_rec(data_list, save_path,phase):
    os.makedirs(save_path, exist_ok=True)
    prefix = os.path.join(save_path,phase)
    record = mx.recordio.MXIndexedRecordIO(prefix + '.idx', prefix + '.rec', 'w')
    for i, (img_path, labe1) in enumerate(tqdm(data_list)):
        img = cv2.imdecode(np.fromfile(img_path), 1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        labe1 = label_enocder(labe1,80)
        p = mx.recordio.pack_img((0, labe1, i, 0), img, quality=95, img_fmt='.jpg')
        record.write_idx(i, p)


if __name__ == '__main__':
    save_path = r'E:\zj\dataset\rec_train'
    data_list = ['E:\\zj\\dataset\\train.txt']
    alphabet = str(np.load('alphabet.npy'))
    label_dict = {}
    for i, char in enumerate(alphabet):
        label_dict[char] = i
    data_list = get_datalist(data_list)
    make_rec(data_list, save_path,'train')
