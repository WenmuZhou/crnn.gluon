#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhoujun
@time: 2019/12/17 下午1:51
'''

from mxnet.gluon import nn


class ConvBlock(nn.HybridBlock):
    def __init__(self, channels, kernel_size):
        super().__init__()
        with self.name_scope():
            self.conv = nn.HybridSequential()
            with self.conv.name_scope():
                self.conv.add(
                    nn.Conv2D(channels, kernel_size, padding=1, use_bias=False),
                    nn.BatchNorm(),
                    nn.LeakyReLU(0.1)
                )

    def hybrid_forward(self, F, x):
        return self.conv(x)


class DownBlock(nn.HybridBlock):
    def __init__(self, channels):
        super().__init__()
        with self.name_scope():
            self.conv = nn.HybridSequential()
            with self.conv.name_scope():
                self.conv.add(
                    ConvBlock(channels, 3),
                    ConvBlock(channels, 3)
                )

    def hybrid_forward(self, F, x):
        return self.conv(x)


class UpBlock(nn.HybridBlock):
    def __init__(self, channels, shrink=True, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.conv3_0 = ConvBlock(channels, 3)
            if shrink:
                self.conv3_1 = ConvBlock(int(channels / 2), 3)
            else:
                self.conv3_1 = ConvBlock(channels, 3)

    def hybrid_forward(self, F, x, s):
        x = F.UpSampling(x, scale=2, sample_type='nearest')

        x = F.Crop(*[x, s], center_crop=True)
        x = F.concat(s, x, dim=1)
        x = self.conv3_0(x)
        x = self.conv3_1(x)
        return x


class UNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super().__init__()
        self.k = kwargs.get('k', 1)
        with self.name_scope():
            self.d0 = DownBlock(32)

            self.d1 = nn.HybridSequential()
            self.d1.add(nn.MaxPool2D(2, 2, ceil_mode=True), DownBlock(64))

            self.d2 = nn.HybridSequential()
            self.d2.add(nn.MaxPool2D(2, 2, ceil_mode=True), DownBlock(128))

            self.d3 = nn.HybridSequential()
            self.d3.add(nn.MaxPool2D(2, 2, ceil_mode=True), DownBlock(256))

            self.d4 = nn.HybridSequential()
            self.d4.add(nn.MaxPool2D(2, 2, ceil_mode=True), DownBlock(512))

            self.u3 = UpBlock(256, shrink=True)
            self.u2 = UpBlock(128, shrink=True)
            self.u1 = UpBlock(64, shrink=True)
            self.u0 = UpBlock(32, shrink=False)

            self.conv = nn.Conv2D(1, 1, use_bias=False)

    def hybrid_forward(self, F, x):
        x0 = self.d0(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)

        y3 = self.u3(x4, x3)
        y2 = self.u2(y3, x2)
        y1 = self.u1(y2, x1)
        y0 = self.u0(y1, x0)

        out = self.conv(y0)
        out = F.sigmoid(out * self.k)
        return out


if __name__ == '__main__':
    import mxnet as mx
    from mxnet import nd

    ctx = mx.cpu()
    net = UNet()
    # net.hybridize()
    net.initialize(ctx=ctx)
    a = nd.zeros((2, 3, 32, 320), ctx=ctx)
    b = net(a)
    print(b.shape)
