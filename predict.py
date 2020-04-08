# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:21
# @Author  : zhoujun
import os
import cv2
import numpy as np
import mxnet as mx
from mxnet import nd, gluon

from data_loader import get_transforms


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
        info = pickle.load(open(model_path.replace('.params', '.info'), 'rb'))
        print('load {} epoch params'.format(info['epoch']))
        config = info['config']
        alphabet = config['dataset']['alphabet']
        self.ctx = try_gpu(gpu_id)

        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)

        self.gpu_id = gpu_id
        img_h, img_w = 32, 100
        for process in config['dataset']['train']['dataset']['args']['pre_processes']:
            if process['type'] == "Resize":
                img_h = process['args']['img_h']
                img_w = process['args']['img_w']
                break
        self.img_w = img_w
        self.img_h = img_h
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        self.alphabet = alphabet
        self.net = get_model(len(alphabet), self.ctx, config['arch']['args'])
        self.net.load_parameters(model_path, self.ctx)
        # self.net = gluon.SymbolBlock.imports('crnn_lite-symbol.json', ['data'], 'crnn_lite-0000.params', ctx=self.ctx)
        self.net.hybridize()

    def predict(self, img_path, model_save_path=None):
        """
        对传入的图像进行预测，支持图像地址和numpy数组
        :param img_path: 图像地址
        :return:
        """
        assert os.path.exists(img_path), 'file is not exists'
        img = self.pre_processing(img_path)
        tensor = self.transform(img)
        tensor = tensor.expand_dims(axis=0)

        tensor = tensor.as_in_context(self.ctx)
        preds, nd_img = self.net(tensor)

        preds = preds.softmax().asnumpy()
        # result = decode(preds, self.alphabet, raw=True)
        # print(result)
        result = decode(preds, self.alphabet)
        print(result)
        if model_save_path is not None:
            # 输出用于部署的模型
            self.net.export(model_save_path)
        return result, img

    def pre_processing(self, img_path):
        """
        对图片进行处理，先按照高度进行resize，resize之后如果宽度不足指定宽度，就补黑色像素，否则就强行缩放到指定宽度
        :param img_path: 图片地址
        :return:
        """
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ratio_h = float(self.img_h) / h
        new_w = int(w * ratio_h)

        if new_w < self.img_w:
            img = cv2.resize(img, (new_w, self.img_h))
            step = np.zeros((self.img_h, self.img_w - new_w, img.shape[-1]), dtype=img.dtype)
            img = np.column_stack((img, step))
        else:
            img = cv2.resize(img, (self.img_w, self.img_h))
        img = nd.array(img)
        return img


if __name__ == '__main__':
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    from models import get_model
    import pickle
    import time
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname=r"msyh.ttc", size=14)

    img_path = '/media/zj/资料/zj/dataset/test_crnn/val/0_song5_0_3.jpg'
    model_path = 'output/crnn_None_CNN_lite_RNN_CTC/checkpoint/model_best.params'

    gluon_net = GluonNet(model_path=model_path, gpu_id=0)
    start = time.time()
    for i in range(10):
        result, img = gluon_net.predict(img_path)
    print((time.time() - start)/10)

    # 输出用于部署的模型
    # gluon_net.net.export('./output/txt4')

    label = result[0][0]
    plt.title(label, fontproperties=font)
    plt.imshow(img.asnumpy().squeeze(), cmap='gray')
    plt.show()
