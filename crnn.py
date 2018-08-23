# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 20:17
# @Author  : zhoujun

import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn, HybridBlock, Block


class Encoder(Block):
    def __init__(self):
        super(Encoder, self).__init__()
        with self.name_scope():
            self.features = nn.Sequential()
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

    def forward(self, x):
        return self.features(x)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.features(x)


class Decoder(Block):
    def __init__(self, hidden_size=256, num_layers=1, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        with self.name_scope():
            self.lstm = mx.gluon.rnn.LSTM(hidden_size, num_layers, bidirectional=True)

    def forward(self, x):
        b, c, h, w = x.shape
        assert h == 1
        x = x.reshape((b, -1, w))
        x = x.transpose((2, 0, 1))# (TNC)(width, batch, channel)
        # x = x.transpose((0, 3, 1, 2))  # (batch,channel,height,width)->(batch,width,channel,height)
        # print('rnn:', x.shape)
        # x = x.flatten()
        # print('rnn:', x.shape)
        # x = x.split(num_outputs=32, axis=1)  # (TNC)(width, batch, channel)
        # # print('rnn:', x.shape)
        # x = nd.concat(*[elem.expand_dims(axis=0) for elem in x], dim=0)
        print('rnn:', x.shape)
        x = self.lstm(x)
        print('rnn:', x.shape)
        x = x.transpose((1, 0, 2))  # (batch, width, channel)
        print('rnn:', x.shape)
        return x

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = x.transpose((0, 3, 1, 2))
        x = x.flatten()
        x = x.split(num_outputs=32, axis=1)  # (SEQ_LEN, N, CHANNELS)
        x = nd.concat(*[elem.expand_dims(axis=0) for elem in x], dim=0)
        x = self.lstm(x)
        x = x.transpose((1, 0, 2))  # (N, SEQ_LEN, HIDDEN_UNITS)
        return x


class CRNN(Block):
    def __init__(self, n_class, hidden_size=256, num_layers=1, **kwargs):
        super(CRNN, self).__init__(**kwargs)
        with self.name_scope():
            self.cnn = Encoder()
            self.rnn = Decoder(hidden_size, num_layers)
            self.fc = nn.Dense(units=n_class, flatten=False)

    def forward(self, x):
        x = self.cnn(x)
        print(x.shape)
        x = self.rnn(x)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        return x

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.cnn(x)
        x = self.rnn(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    ctx = mx.gpu(0)
    a = nd.zeros((2, 3, 32, 320), ctx=ctx)
    net = CRNN(10, 256, 1)
    net.initialize(ctx=ctx)
    # net.hybridize()
    print(net)
    b = net(a)
    print(b.shape)
