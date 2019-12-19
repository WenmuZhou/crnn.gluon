#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhoujun
@time: 2019/12/19 下午3:18
'''
import imgaug.augmenters as iaa


class IaaAugment():
    def __init__(self):
        self.seq = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(2, 7)),  # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 11)),  # blur image using local medians with kernel sizes between 2 and 7
            ])),
            iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
            iaa.Sometimes(0.5, iaa.Add((-10, 10), per_channel=0.5)),
            iaa.Sometimes(0.5, iaa.AddToHueAndSaturation((-20, 20))),
            iaa.Sometimes(0.5, iaa.FrequencyNoiseAlpha(
                exponent=(-4, 0),
                first=iaa.Multiply((0.5, 1.5), per_channel=True),
                second=iaa.LinearContrast((0.5, 2.0))
            )),
            iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.05))),
            iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1)))
        ], random_order=True)

    def __call__(self, img):
        return self.seq.augment_image(img)
