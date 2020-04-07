#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhoujun
@time: 2019/12/17 下午1:51
'''
from models.modules.seg.resnet_fpn import ResNetFPN
from models.modules.seg.unet import UNet

if __name__ == '__main__':
    import numpy as np
    import time
    import mxnet as mx

    ctx = mx.gpu(0)
    net = ResNetFPN(backbone='resnet34_v1b', channels=1, ctx=ctx, pretrained=True)
    # net = UNet()
    net.initialize(ctx=ctx)
    net.hybridize()
    x = mx.nd.array(np.random.uniform(-2, 4.2, size=(64, 3, 32, 320)), ctx=ctx)

    for i in range(10):
        tic = time.time()
        y = net(x)
        print(time.time() - tic)
    print(y.shape)
