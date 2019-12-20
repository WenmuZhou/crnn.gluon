from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.model_zoo.vision.resnet import BasicBlockV2
from mxnet.gluon.model_zoo.vision.densenet import _make_dense_block


class VGG(HybridBlock):
    def __init__(self, **kwargs):
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


class ResNet(HybridBlock):
    def __init__(self, **kwargs):
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
                    BasicBlockV2(64, 1, True),
                    BasicBlockV2(128, 1, True),
                    nn.Dropout(0.2),

                    BasicBlockV2(128, 2, True),
                    BasicBlockV2(256, 1, True),
                    nn.Dropout(0.2),

                    nn.Conv2D(256, 2, strides=(2, 1), padding=(0, 1), use_bias=False),

                    BasicBlockV2(512, 1, True),
                    nn.BatchNorm(),
                    nn.Activation('relu'),

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


class DenseNet(HybridBlock):
    def __init__(self, **kwargs):
        super(DenseNet, self).__init__()
        with self.name_scope():
            self.features = nn.HybridSequential()
            self.features.add(nn.Conv2D(64, 5, padding=2, strides=2, use_bias=False))
            self.features.add(_make_dense_block(8, 4, 8, 0, 1))
            self.features.add(_make_transition(128, 2, 0, 0.2))

            self.features.add(_make_dense_block(8, 4, 8, 0, 2))
            self.features.add(_make_transition(128, (2, 1), 0, 0.2))

            self.features.add(_make_dense_block(8, 4, 8, 0, 3))

            self.features.add(
                nn.BatchNorm(),
                nn.Activation('relu')
            )

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = x.reshape((0, -1, 1, 0))
        return x


if __name__ == '__main__':
    import time
    from mxnet import nd
    import mxnet as mx
    from models.modules.seg import UNet, ResNetFPN

    ctx = mx.gpu(0)
    input = nd.random.uniform(2, 4, (1, 3, 32, 320), ctx=ctx)
    net = nn.HybridSequential()
    # net.add(UNet())
    net.add(VGG())
    net.hybridize()
    net.initialize(ctx=ctx)

    tic = time.time()
    for i in range(1):
        y = net(input)
    all_time = time.time() - tic
    fps = (input.shape[0] * 1) / all_time
    print('batch image time: {},fps: {}'.format(all_time / 10, fps))
    print(y.shape)
