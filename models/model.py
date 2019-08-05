from mxnet import nd
from mxnet.gluon import nn, HybridBlock
from models.modules.feature_extraction import VGG, ResNet, DenseNet
from models.modules.sequence_modeling import Decoder


class Model(HybridBlock):
    def __init__(self, n_class, config):
        super(Model, self).__init__()
        feature_extraction_dict = {'VGG': VGG, 'ResNet': ResNet, 'DenseNet': DenseNet}

        # 特征提取模型设置
        feature_extraction_type = config['feature_extraction']['type']
        if feature_extraction_type in feature_extraction_dict.keys():
            self.feature_extraction = feature_extraction_dict[feature_extraction_type]()
        else:
            raise NotImplementedError

        # 序列模型
        sequence_model_type = config['sequence_model']['type']
        if sequence_model_type == 'RNN':
            self.sequence_model = Decoder(config['sequence_model']['args']['hidden_size'])
        else:
            self.sequence_model = None

        # 预测设置
        self.prediction_type = config['prediction']['type']
        if self.prediction_type == 'CTC':
            self.prediction = nn.Dense(units=n_class, flatten=False)
        else:
            raise NotImplementedError
        self.model_name = '{}_{}_{}'.format(feature_extraction_type, sequence_model_type,
                                            self.prediction_type)

        self.batch_max_length = -1

    def get_batch_max_length(self, img_h, img_w, ctx):
        input = nd.zeros((2, 3, img_h, img_w), ctx=ctx)
        # 特征提取阶段
        visual_feature = self.feature_extraction(input)
        self.batch_max_length = visual_feature.shape[-1]
        return self.batch_max_length

    def hybrid_forward(self, F, x, *args, **kwargs):
        # 特征提取阶段
        visual_feature = self.feature_extraction(x)
        visual_feature = visual_feature.squeeze(axis=2)
        visual_feature = visual_feature.transpose((0, 2, 1))  # (NTC)(batch, width, channel)s
        # 序列建模阶段
        if self.sequence_model is not None:
            contextual_feature = self.sequence_model(visual_feature)
        else:
            contextual_feature = visual_feature
        # 预测阶段
        if self.prediction_type == 'CTC':
            prediction = self.prediction(contextual_feature)
        else:
            raise NotImplementedError
        return prediction


if __name__ == '__main__':
    import os
    import mxnet as mx
    import numpy as np
    from utils import read_json

    config = read_json(r'E:\zj\code\crnn.gluon\config.json')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config['trainer']['gpus']])

    config['data_loader']['args']['alphabet'] = str(np.load(r'E:\zj\code\crnn.gluon\alphabet.npy'))
    alphabet = config['data_loader']['args']['alphabet']
    # checkpoint = torch.load(config['trainer']['resume']['checkpoint'])
    ctx = mx.cpu()
    net = Model(len(alphabet), config['arch']['args'])
    net.hybridize()
    net.initialize(ctx=ctx)
    print(net.model_name)
    print(net.get_batch_max_length(32,320,ctx))
    # a = nd.zeros((2, 3, 32, 320), ctx=ctx)
    # b = net(a)
    # print(b.shape)
