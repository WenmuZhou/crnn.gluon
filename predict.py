# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:21
# @Author  : zhoujun

import os
import keys
import mxnet as mx
from mxnet import image, nd
from mxnet.gluon.data.vision import transforms
from crnn import CRNN


def try_gpu(gpu):
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu(gpu)
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


def decode(preds, alphabet, raw=False):
    results = []
    alphabet_size = len(alphabet)
    for word in preds:
        if raw:
            results.append(''.join([alphabet[int(i)] for i in word]))
        else:
            result = []
            for i, index in enumerate(word):
                if i < len(word) - 1 and word[i] == word[i + 1] and word[-1] != -1:  # Hack to decode label as well
                    continue
                if index == -1 or index == alphabet_size - 1:
                    continue
                else:
                    result.append(alphabet[int(index)])
            results.append(''.join(result))
    return results


class GluonNet:
    def __init__(self, model_path, alphabet, img_shape, net, img_channel=3, gpu_id=None):
        """
        初始化gluon模型
        :param model_path: 模型地址
        :param alphabet: 字母表
        :param img_shape: 图像的尺寸(w,h)
        :param net: 网络计算图，如果在model_path中指定的是参数的保存路径，则需要给出网络的计算图
        :param img_channel: 图像的通道数: 1,3
        :param gpu_id: 在哪一块gpu上运行
        """
        self.gpu_id = gpu_id
        self.img_w = img_shape[0]
        self.img_h = img_shape[1]
        self.img_channel = img_channel
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
        img = transforms.ToTensor()(img)
        img = img.expand_dims(axis=0)

        img = img.as_in_context(self.ctx)
        preds = self.net(img)

        preds = preds.softmax().topk(axis=2).asnumpy()
        result = decode(preds, self.alphabet, raw=True)
        print(result)
        result = decode(preds, self.alphabet)
        print(result)
        return result

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
        if new_w < self.img_w:
            step = nd.zeros((self.img_h, self.img_w - new_w, 3), dtype=img.dtype)
            img = nd.concat(img, step, dim=1)
        return img


if __name__ == '__main__':
    import time
    from mxnet import gluon
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname=r"simsun.ttc", size=14)

    img_path = '/data/zhy/crnn/Chinese_character/data/上海医院名称合并_20180625/HP31011210017400_上海交通大学医学院附属仁济医院南院_f202017919125659 (2)_0_ori_上海交通大学医学院附属仁济医院南院.jpg'
    # model_path = 'output/crnn_lstm_all/53_0.9920078039430449.params'
    # net = CRNN(len(keys.all_alphabet), hidden_size=256)
    # gluon_net = GluonNet(model_path=model_path, alphabet=keys.all_alphabet, img_shape=(320, 32), img_channel=3, net=net,
    #                      gpu_id=0)
    # start = time.time()
    # result = gluon_net.predict(img_path)
    # print(time.time() - start)
    #
    # gluon_net.net.export('./all')
    img_h = 32
    img_w = 320
    img = image.imdecode(open(img_path, 'rb').read(), 1)
    h, w = img.shape[:2]
    ratio_h = float(img_h) / h
    new_w = int(w * ratio_h)
    img = image.imresize(img, w=new_w, h=img_h)
    if new_w < img_w:
        step = nd.zeros((img_h, img_w - new_w, 3), dtype=img.dtype)
        img = nd.concat(img, step, dim=1)

    img = transforms.ToTensor()(img)
    img = img.expand_dims(axis=0)
    ctx = try_gpu(0)
    # img = img.as_in_context(ctx)
    net = gluon.SymbolBlock.imports('all-symbol.json',['data'],'all-0000.params')
    start = time.time()
    result = net(img)

    result = result.softmax().topk(axis=2).asnumpy()
    result = decode(result, keys.all_alphabet)
    label = result[0]
    print(time.time() - start)
    img = image.imread(img_path)
    img = img.asnumpy()
    plt.title(label, fontproperties=font)
    plt.imshow(img)
    plt.show()
