#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhoujun
@time: 2019/12/19 下午2:56
'''

from mxnet.gluon import nn, HybridBlock


class CTC(HybridBlock):
    def __init__(self, n_class, **kwargs):
        super().__init__()
        self.fc = nn.Dense(units=n_class, flatten=False)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.fc(x)