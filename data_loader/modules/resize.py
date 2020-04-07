#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhoujun
@time: 2019/12/19 下午3:23
'''
import cv2
import numpy as np
from mxnet import image, nd


class Resize:
    def __init__(self, img_h, img_w, pad=True, **kwargs):
        self.img_h = img_h
        self.img_w = img_w
        self.pad = pad

    def __call__(self, img: np.ndarray):
        """
        对图片进行处理，先按照高度进行resize，resize之后如果宽度不足指定宽度，就补黑色像素，否则就强行缩放到指定宽度
        :param img_path: 图片地址
        :return:
        """
        assert len(img.shape) == 3 and img.shape[-1] in [1, 3]
        h, w = img.shape[:2]
        ratio_h = self.img_h / h
        new_w = int(w * ratio_h)
        if new_w < self.img_w and self.pad:
            img = cv2.resize(img, (new_w, self.img_h))
            step = np.zeros((self.img_h, self.img_w - new_w, img.shape[-1]), dtype=img.dtype)
            img = np.column_stack((img, step))
        else:
            img = cv2.resize(img, (self.img_w, self.img_h))
        return img


class ResizeRandomCrop:
    def __init__(self, img_h, img_w, pad=True, **kwargs):
        self.img_h = img_h
        self.img_w = img_w
        self.pad = pad
        self.phase = kwargs['phase']

    def __call__(self, img: np.ndarray):
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
        h, w = img.shape[:2]
        ratio_h = float(img_h) / h
        new_w = int(w * ratio_h)
        if new_w < img_w and self.pad:
            img = cv2.resize(img, (new_w, img_h))
            if len(img.shape) == 2:
                img = np.expand_dims(img, 3)
            step = np.zeros((img_h, img_w - new_w, img.shape[-1]), dtype=img.dtype)
            img = np.column_stack((img, step))
        else:
            img = cv2.resize(img, (img_w, img_h))
        if data_augment:
            img = nd.array(img)
            img, _ = image.random_crop(img, (self.img_w, self.img_h))
            img = img.asnumpy()
        return img
