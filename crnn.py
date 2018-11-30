# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 20:17
# @Author  : zhoujun

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.model_zoo.vision.resnet import BasicBlockV2
from mxnet.gluon.model_zoo.vision.densenet import _make_dense_block


class VGG(nn.HybridBlock):
    def __init__(self):
        super(VGG, self).__init__()
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


class ResNet(nn.HybridBlock):
    def __init__(self):
        super(ResNet, self).__init__()
        with self.name_scope():
            self.features = nn.HybridSequential()
            with self.features.name_scope():
                self.features.add(
                    nn.Conv2D(64, 3, padding=1, use_bias=False),
                    nn.BatchNorm(),
                    nn.Activation('relu'),
                    # nn.MaxPool2D(pool_size=2, strides=2),
                    nn.Conv2D(64, 2, strides=2, use_bias=False),
                    nn.BatchNorm(),
                    nn.Activation('relu'),
                    BasicBlockV2(64, 1, True),
                    BasicBlockV2(128, 1, True),
                    nn.Dropout(0.2),

                    BasicBlockV2(128, 2, True),
                    BasicBlockV2(256, 1, True),
                    nn.Dropout(0.2),

                    nn.Conv2D(256, 2, strides=(2, 1), padding=(0, 1), use_bias=False),

                    BasicBlockV2(512, 1, True),

                    nn.Conv2D(1024, 3, padding=0, use_bias=False),
                    nn.BatchNorm(),
                    nn.Activation('relu'),
                    nn.Conv2D(2048, 2, padding=(0, 1), use_bias=False),
                    nn.BatchNorm(),
                    nn.Activation('relu'),
                )

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.features(x)


def _make_transition(num_output_features, pool_stride, pool_pad, dropout):
    out = nn.HybridSequential(prefix='')
    out.add(nn.BatchNorm())
    out.add(nn.Activation('relu'))
    out.add(nn.Conv2D(num_output_features, kernel_size=1, use_bias=False))
    if dropout:
        out.add(nn.Dropout(dropout))
    out.add(nn.AvgPool2D(pool_size=2, strides=pool_stride, padding=pool_pad))
    return out


class DenseNet(nn.HybridBlock):
    def __init__(self):
        super(DenseNet, self).__init__()
        with self.name_scope():
            self.features = nn.HybridSequential()
            self.features.add(
                nn.Conv2D(64, 3, padding=1, use_bias=False),
                nn.BatchNorm(),
                nn.Activation('relu'),
                # nn.MaxPool2D(pool_size=2, strides=2),
                nn.Conv2D(64, 2, strides=2, use_bias=False),
            )
            self.features.add(_make_dense_block(8, 4, 8, 0, 1))
            self.features.add(_make_transition(128, 2, 0, 0.2))

            self.features.add(_make_dense_block(8, 4, 8, 0, 2))
            self.features.add(_make_transition(192, (2, 1), (0, 1), 0.2))

            self.features.add(_make_dense_block(8, 4, 8, 0, 3))

            self.features.add(
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(512, 3, padding=0, use_bias=False),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(1024, 2, padding=(0, 1), use_bias=False),
                nn.BatchNorm(),
                nn.Activation('relu')
            )

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.features(x)


class BidirectionalGRU(HybridBlock):
    def __init__(self, hidden_size, num_layers, nOut):
        super(BidirectionalGRU, self).__init__()
        with self.name_scope():
            self.rnn = mx.gluon.rnn.GRU(hidden_size, num_layers, bidirectional=True, layout='NTC')
            self.fc = nn.Dense(units=nOut, flatten=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.rnn(x)
        x = self.fc(x)  # [T * b, nOut]
        return x


class BidirectionalLSTM(HybridBlock):
    def __init__(self, hidden_size, num_layers, nOut):
        super(BidirectionalLSTM, self).__init__()
        with self.name_scope():
            self.rnn = mx.gluon.rnn.LSTM(hidden_size, num_layers, bidirectional=True, layout='NTC')
            # self.fc = nn.Dense(units=nOut, flatten=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.rnn(x)
        # x = self.fc(x)  # [T * b, nOut]
        return x


class Encoder(HybridBlock):
    def __init__(self):
        super(Encoder, self).__init__()
        with self.name_scope():
            self.features = ResNet()#DenseNet()  # VGG()

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.features(x)


class Decoder(HybridBlock):
    def __init__(self, n_class, hidden_size=256, num_layers=1):
        super(Decoder, self).__init__()
        with self.name_scope():
            self.lstm = nn.HybridSequential()
            with self.lstm.name_scope():
                self.lstm.add(BidirectionalLSTM(hidden_size, num_layers, hidden_size * 2))
                self.lstm.add(BidirectionalLSTM(hidden_size, num_layers, n_class))
            # self.lstm1 = BidirectionalLSTM(hidden_size, num_layers, hidden_size * 2)
            # self.lstm2 = BidirectionalLSTM(hidden_size, num_layers, n_class)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # y = self.lstm1(x)
        # x = x + y
        # y = self.lstm2(x)
        # x = x + y
        x = self.lstm(x)
        return x


class CRNN(HybridBlock):
    def __init__(self, n_class, hidden_size=256, num_layers=1):
        super(CRNN, self).__init__()
        with self.name_scope():
            self.cnn = Encoder()
            self.rnn = Decoder(n_class, hidden_size, num_layers)
            self.fc = nn.Dense(units=n_class, flatten=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.cnn(x)
        x = x.squeeze(axis=2)
        x = x.transpose((0, 2, 1))  # (NTC)(batch, width, channel)
        x = self.rnn(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    print(mx.__version__)
    ctx = mx.cpu()
    a = nd.zeros((2, 3, 32, 320), ctx=ctx)
    net = CRNN(10,256)
    # net = VGG()
    # net.hybridize()
    net.initialize(ctx=ctx)
    b = net(a)
    print(b.shape)
    print(net)
    # print(net.summary(a))
