#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhoujun
@time: 2019/12/17 上午11:02
'''
from mxnet.gluon import HybridBlock, nn
import mxnet as mx
import gluoncv.model_zoo as gcv_model_zoo
from gluoncv.nn.feature import FPNFeatureExpander


class ResNetFPN(HybridBlock):
    def __init__(self, backbone, channels=1, ctx=mx.cpu(), pretrained=False, **kwargs):
        super().__init__()
        self.k = kwargs.get('k', 1)
        self.channels = channels
        model_dict = {'resnet18_v1b': ['layers1_relu3_fwd', 'layers2_relu3_fwd', 'layers3_relu3_fwd', 'layers4_relu3_fwd'],
                      'resnet34_v1b': ['layers1_relu5_fwd', 'layers2_relu7_fwd', 'layers3_relu11_fwd', 'layers4_relu3_fwd']}
        backbone_model = getattr(gcv_model_zoo, backbone)

        backbone_outputs = model_dict[backbone]
        base_network = backbone_model(pretrained=pretrained, norm_layer=nn.BatchNorm, ctx=ctx, **kwargs)
        self.features = FPNFeatureExpander(
            network=base_network,
            outputs=backbone_outputs, num_filters=[256, 256, 256, 256], use_1x1=True,
            use_upsample=True, use_elewadd=True, use_p6=False, no_bias=True, pretrained=pretrained,
            ctx=ctx)

        self.extrac_convs = []

        for i in range(4):
            weight_init = mx.init.Normal(0.001)
            extra_conv = nn.HybridSequential(prefix='extra_conv_{}'.format(i))
            with extra_conv.name_scope():
                extra_conv.add(nn.Conv2D(256, 3, 1, 1))
                extra_conv.add(nn.BatchNorm())
                extra_conv.add(nn.Activation('relu'))
            extra_conv.initialize(weight_init, ctx=ctx)
            self.register_child(extra_conv)
            self.extrac_convs.append(extra_conv)

        self.decoder_out = nn.HybridSequential(prefix='decoder_out')
        with self.decoder_out.name_scope():
            weight_init = mx.init.Normal(0.001)
            self.decoder_out.add(nn.Conv2D(256, 3, 1, 1))
            self.decoder_out.add(nn.BatchNorm())
            self.decoder_out.add(nn.Activation('relu'))
            self.decoder_out.add(nn.Conv2D(self.channels, 1, 1))
            self.decoder_out.initialize(weight_init, ctx=ctx)

    def hybrid_forward(self, F, x, **kwargs):
        # output: c4 -> c1 [1/4, 1/8, 1/16. 1/32]
        fpn_features = self.features(x)

        concat_features = []
        scales = [1, 2, 4, 8]
        for i, C in enumerate(fpn_features):
            extrac_C = self.extrac_convs[i](C)
            up_C = F.UpSampling(extrac_C, scale=scales[i], sample_type='nearest', name="extra_upsample_{}".format(i))
            concat_features.append(up_C)
        concat_output = F.concat(*concat_features, dim=1)
        output = self.decoder_out(concat_output)
        output = F.sigmoid(output * self.k)
        output = F.UpSampling(output, scale=4, sample_type='nearest', name="final_upsampling")
        return output
