# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:21
# @Author  : zhoujun

import os
import numpy as np
import mxnet as mx
from mxnet import image, nd
from mxnet.gluon.data.vision import transforms


def try_gpu(gpu):
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu(gpu)
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


def decode(preds, alphabet, raw=False):
    if len(preds.shape) > 2:
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
    else:
        preds_idx = preds
        preds_prob = np.ones_like(preds)
    result_list = []
    alphabet_size = len(alphabet)
    for word, prob in zip(preds_idx, preds_prob):
        if raw:
            result_list.append((''.join([alphabet[int(i)] for i in word]), prob))
        else:
            result = []
            conf = []
            for i, index in enumerate(word):
                if i < len(word) - 1 and word[i] == word[i + 1] and word[-1] != -1:  # Hack to decode label as well
                    continue
                if index == -1 or index == alphabet_size - 1:
                    continue
                else:
                    result.append(alphabet[int(index)])
                    conf.append(prob[i])
            result_list.append((''.join(result), conf))
    return result_list


class GluonNet:
    def __init__(self, model_path, gpu_id=None):
        """
        初始化gluon模型
        :param model_path: 模型地址
        :param gpu_id: 在哪一块gpu上运行
        """
        config = pickle.load(open(model_path.replace('.params', '.info'),'rb'))['config']
        alphabet = config['data_loader']['args']['dataset']['alphabet']
        net = get_model(len(alphabet), config['arch']['args'])

        self.gpu_id = gpu_id
        self.img_w = config['data_loader']['args']['dataset']['img_w']
        self.img_h = config['data_loader']['args']['dataset']['img_h']
        self.img_channel = config['data_loader']['args']['dataset']['img_channel']
        self.alphabet = alphabet
        self.ctx = try_gpu(gpu_id)
        self.net = net
        self.net.load_parameters(model_path, self.ctx)
        self.net.hybridize()

    def predict(self, img_path):
        """
        对传入的图像进行预测，支持图像地址和numpy数组
        :param img_path: 图像地址
        :return:
        """
        assert self.img_channel in [1, 3], 'img_channel must in [1.3]'
        assert os.path.exists(img_path), 'file is not exists'
        img = self.pre_processing(img_path)
        img1 = transforms.ToTensor()(img)
        img1 = img1.expand_dims(axis=0)

        img1 = img1.as_in_context(self.ctx)
        preds = self.net(img1)

        preds = preds.softmax().asnumpy()
        # result = decode(preds, self.alphabet, raw=True)
        # print(result)
        result = decode(preds, self.alphabet)
        print(result)
        return result, img

    def pre_processing(self, img_path):
        """
        对图片进行处理，先按照高度进行resize，resize之后如果宽度不足指定宽度，就补黑色像素，否则就强行缩放到指定宽度
        :param img_path: 图片地址
        :return:
        """
        img = image.imdecode(open(img_path, 'rb').read(), 1 if self.img_channel == 3 else 0)
        h, w = img.shape[:2]
        ratio_h = float(self.img_h) / h
        new_w = int(w * ratio_h)
        img = image.imresize(img, w=new_w, h=self.img_h)
        # if new_w < self.img_w:
        #     step = nd.zeros((self.img_h, self.img_w - new_w, self.img_channel), dtype=img.dtype)
        #     img = nd.concat(img, step, dim=1)
        return img


if __name__ == '__main__':
    from models import get_model
    import pickle
    import time
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname=r"msyh.ttc", size=14)

    img_path = 'E:/zj/dataset/train/0_song5_0_3_w.jpg'
    model_path = 'output/crnn_VGG_RNN_CTC/checkpoint/CRNN_2_loss1.602779_val_acc1.000000.params'


    gluon_net = GluonNet(model_path=model_path, gpu_id=None)
    start = time.time()
    result, img = gluon_net.predict(img_path)
    print(time.time() - start)

    # 输出用于部署的模型
    # gluon_net.net.export('./output/txt4')

    label = result[0][0]
    plt.title(label, fontproperties=font)
    plt.imshow(img.asnumpy().squeeze(), cmap='gray')
    plt.show()
