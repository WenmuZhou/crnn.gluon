# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:18
# @Author  : zhoujun

from __future__ import print_function

import os, time
from PIL import Image
import numpy as np
import mxnet as mx
import random, cv2
from mxnet.gluon.data.vision import ImageFolderDataset


class Gluon_OCRIter(ImageFolderDataset):
    def __init__(self, data_root, data_list, data_shape, num_label):
        # self._root = os.path.expanduser(root)
        # self._flag = flag
        # self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']

        self.data_root = data_root
        self.data_list = data_list
        self.num_label = num_label
        self.data_shape = data_shape
        self._list_images(self.data_root, self.data_list)

    def _list_images(self, data_root, data_list):

        self.synsets = []
        self.items = []
        self.dataset_lines = open(data_list).readlines()
        start_ts = time.time()
        cnt = 0
        for m_line in self.dataset_lines:
            img_lst = m_line.strip().split(' ')
            # img_path = os.path.join(self.data_root, img_lst[0])
            # if not os.path.exists(img_path):
            #    continue
            # cnt += 1
            # ret = np.zeros(self.num_label, dtype=np.float32) -1
            # for idx in range(1, len(img_lst)):
            #    ret[idx - 1] = int(img_lst[idx])
            # self.items.append((img_path,ret))
            self.items.append(img_lst)
        print("list images take %f seconds" % (time.time() - start_ts))

    def __getitem__(self, idx):
        # img = image.imread(self.items[idx][0], self._flag)
        img_lst = self.items[idx]
        img_path = os.path.join(self.data_root, img_lst[0])
        label = np.zeros(self.num_label, dtype=np.float32) - 1
        for idx_ in range(1, len(img_lst)):
            label[idx_ - 1] = int(img_lst[idx_]) - 1
        # label = self.items[idx][1]
        DESIRED_SIZE = (self.data_shape[0], self.data_shape[1])
        img = self.pre_processing(img_path, DESIRED_SIZE)
        img = np.array(img).reshape((1, self.data_shape[0], self.data_shape[1])).astype(np.float32)
        # if self._transform is not None:
        #     return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self.items)

    def pre_processing(self, img_in, DESIRED_SIZE):
        # im =cv2.imread(img_in)
        im = cv2.imread(img_in, cv2.IMREAD_GRAYSCALE)
        size = im.shape[:2]  # old_size is in (height, width) format
        if size[0] > DESIRED_SIZE[0] or size[1] > DESIRED_SIZE[1]:
            ratio_w = float(DESIRED_SIZE[0]) / size[0]
            ratio_h = float(DESIRED_SIZE[1]) / size[1]
            ratio = min(ratio_w, ratio_h)
            new_size = tuple([int(x * ratio) for x in size])
            im = cv2.resize(im, (new_size[1], new_size[0]))
            size = im.shape

        delta_w = max(0, DESIRED_SIZE[1] - size[1])
        delta_h = max(0, DESIRED_SIZE[0] - size[0])
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = im[0][0]
        if color < 230:
            color = 230
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=float(color))
        # new_im = np.asarray(new_im)
        return new_im
