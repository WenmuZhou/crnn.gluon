from mxnet import nd
from mxnet.gluon import HybridBlock
from models.modules import *


def init_modules(config, module_name, canbe_none=True, **kwargs):
    if module_name not in config:
        return None, None
    module_config = config[module_name]
    module_type = module_config['type']
    if len(module_type) == 0:
        return None, None
    if 'args' not in module_config or module_config['args'] is None:
        module_args = {}
    else:
        module_args = module_config['args']
    module_args.update(**kwargs)
    if canbe_none:
        try:
            module = eval(module_type)(**module_args)
        except:
            module = None
    else:
        module = eval(module_type)(**module_args)
    return module, module_type


class Model(HybridBlock):
    def __init__(self, n_class, ctx, config):
        super(Model, self).__init__()

        # 二值分割网络
        self.binarization, self.binarization_type = init_modules(config, 'binarization', canbe_none=True, ctx=ctx)

        # 特征提取模型设置
        self.feature_extraction, self.feature_extraction_type = init_modules(config, 'feature_extraction', canbe_none=False)

        # 序列模型
        self.sequence_model, self.sequence_model_type = init_modules(config, 'sequence_model', canbe_none=True)

        # 预测设置
        self.prediction, self.prediction_type = init_modules(config, 'prediction', canbe_none=False, n_class=n_class)

        self.model_name = '{}_{}_{}_{}'.format(self.binarization_type, self.feature_extraction_type, self.sequence_model_type, self.prediction_type)
        self.batch_max_length = -1

    def get_batch_max_length(self, x):
        # 特征提取阶段
        if self.binarization is not None:
            x = self.binarization(x)
        visual_feature = self.feature_extraction(x)
        self.batch_max_length = visual_feature.shape[-1]
        return self.batch_max_length

    def hybrid_forward(self, F, x, *args, **kwargs):
        if self.binarization is not None:
            x = self.binarization(x)
        # 特征提取阶段
        visual_feature = self.feature_extraction(x)
        # 序列建模阶段
        if self.sequence_model is not None:
            contextual_feature = self.sequence_model(visual_feature)
        else:
            contextual_feature = visual_feature.squeeze(axis=2).transpose((0, 2, 1))
        # 预测阶段
        if 'CTC' in self.prediction_type:
            prediction = self.prediction(contextual_feature)
        else:
            raise NotImplementedError
        return prediction, x


if __name__ == '__main__':
    import anyconfig
    import mxnet as mx
    import numpy as np
    from utils import parse_config

    ctx = mx.cpu()
    a = nd.random.randn(2, 3, 32, 320, ctx=ctx)
    config = anyconfig.load(open(r'E:\zj\code\crnn.gluon\config\icdar2015.yaml', 'rb'))
    if 'base' in config:
        config = parse_config(config)
    config['dataset']['alphabet'] = str(np.load(r'E:\zj\code\crnn.gluon\alphabet.npy'))
    alphabet = config['dataset']['alphabet']
    # checkpoint = torch.load(config['trainer']['resume']['checkpoint'])
    net = Model(len(alphabet),ctx, config['arch']['args'])
    # net.hybridize()
    net.initialize(ctx=ctx)
    print(net.model_name)
    print(net.get_batch_max_length(a))
    b = net(a)[0]
    print(b.shape)
