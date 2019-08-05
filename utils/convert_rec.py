# -*- coding: utf-8 -*-
# @Time    : 18-8-31 上午9:48
# @Author  : zhoujun

import os
import pathlib
from tqdm import tqdm
import numpy as np
import mxnet as mx
import cv2

data_shape = (320, 32)
img_channel = 3
num_label = 80
alphabet = ''
data_root = '/data1/zj/data/crnn/txt'
train_fn = '/data/zhy/crnn/Chinese_character/train2.txt'
val_fn = '/data/zhy/crnn/Chinese_character/test2.txt'

if not os.path.exists(data_root):
    os.makedirs(data_root)
label_dict = {}
for i, char in enumerate(alphabet):
    label_dict[char] = i


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


def label_enocder(label, num_label):
    """
    对label进行处理，将输入的label字符串转换成在字母表中的索引
    :param label: label字符串
    :return: 索引列表
    """
    tmp_label = np.zeros(len(label))
    for i, ch in enumerate(label):
        tmp_label[i] = label_dict[ch]
    return tmp_label


def make_rec(data_txt, prefix, num_label):
    record = mx.recordio.MXIndexedRecordIO(prefix + '.idx', prefix + '.rec', 'w')
    with open(data_txt, 'r', encoding='utf-8') as f:
        i = 0
        lines = f.readlines()
        pbar = tqdm(total=len(lines))
        for line in lines:
            line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
            img_path = pathlib.Path(line[0])
            if img_path.exists() and img_path.stat().st_size > 0 and line[1]:
                img = pre_processing(line[0], data_shape, img_channel)
                label = label_enocder(line[1], num_label)
                p = mx.recordio.pack_img((0, label, i, 0), img, quality=95, img_fmt='.jpg')
                record.write_idx(i, p)
                i += 1
                # if i > 100:
                #     break
            pbar.update(1)
    record.close()
    pbar.close()


make_rec(val_fn, data_root + '/val', num_label)
# make_rec(train_fn, data_root + '/train', num_label)
