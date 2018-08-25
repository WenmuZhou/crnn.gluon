# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 20:17
# @Author  : zhoujun

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn, HybridBlock


class Encoder(HybridBlock):
    def __init__(self):
        super(Encoder, self).__init__()
        with self.name_scope():
            self.features = nn.HybridSequential()
            with self.features.name_scope():
                self.features.add(
                    # conv layer
                    nn.Conv2D(kernel_size=(3, 3), padding=(1, 1), channels=64, activation="relu"),
                    nn.BatchNorm(),
                    nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

                    # second conv layer
                    nn.Conv2D(kernel_size=(3, 3), padding=(1, 1), channels=128, activation="relu"),
                    nn.BatchNorm(),
                    nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

                    # third conv layer
                    nn.Conv2D(kernel_size=(3, 3), padding=(1, 1), channels=256, activation="relu"),
                    nn.BatchNorm(),

                    # fourth conv layer
                    nn.Conv2D(kernel_size=(3, 3), padding=(1, 1), channels=256, activation="relu"),
                    nn.BatchNorm(),
                    nn.MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding=(0, 1)),

                    # fifth conv layer
                    nn.Conv2D(kernel_size=(3, 3), padding=(1, 1), channels=512, activation="relu"),
                    nn.BatchNorm(),

                    # sixth conv layer
                    nn.Conv2D(kernel_size=(3, 3), padding=(1, 1), channels=512, activation="relu"),
                    nn.BatchNorm(),
                    nn.MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding=(0, 1)),

                    # seren conv layer
                    nn.Conv2D(kernel_size=(2, 2), channels=512, activation="relu"),
                    nn.BatchNorm()
                )

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.features(x)


class BidirectionalGRU(HybridBlock):
    def __init__(self, hidden_size, num_layers, nOut):
        super(BidirectionalGRU, self).__init__()
        with self.name_scope():
            self.rnn = mx.gluon.rnn.LSTM(hidden_size, num_layers, bidirectional=True, layout='NTC')
            self.fc = nn.Dense(units=nOut, flatten=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.rnn(x)
        x = self.fc(x)  # [T * b, nOut]
        return x


class Decoder(HybridBlock):
    def __init__(self, n_class, hidden_size=256, num_layers=1, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        with self.name_scope():
            self.lstm = nn.HybridSequential()
            with self.lstm.name_scope():
                self.lstm.add(BidirectionalGRU(hidden_size, num_layers, hidden_size * 2))
                self.lstm.add(BidirectionalGRU(hidden_size, num_layers, n_class))

    def hybrid_forward(self, F, x, *args, **kwargs):
        # b, c, h, w = x.shape
        # assert h == 1
        x = x.squeeze(axis=2)
        x = x.transpose((0, 2, 1))  # (NTC)(batch, width, channel)
        # x = x.transpose((0, 3, 1, 2))
        # x = x.flatten()
        # x = x.split(num_outputs=32, axis=1)  # (SEQ_LEN, N, CHANNELS)
        # x = nd.concat(*[elem.expand_dims(axis=0) for elem in x], dim=0)
        x = self.lstm(x)
        return x


class CRNN(HybridBlock):
    def __init__(self, n_class, hidden_size=256, num_layers=1, **kwargs):
        super(CRNN, self).__init__(**kwargs)
        with self.name_scope():
            self.cnn = Encoder()
            self.rnn = Decoder(n_class, hidden_size, num_layers)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.cnn(x)
        x = self.rnn(x)
        return x


if __name__ == '__main__':
    ctx = mx.cpu(0)
    a = nd.zeros((2, 3, 32, 320), ctx=ctx)
    net = CRNN(10, 256, 1)
    # net.hybridize()
    net.initialize(ctx=ctx)
    b = net(a)
    print(b.shape)
    print(net)
