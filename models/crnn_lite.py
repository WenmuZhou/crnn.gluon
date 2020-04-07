# -*- coding: utf-8 -*-
# @Time    : 2020/4/7 18:31
# @Author  : zhoujun
import mxnet as mx
from mxnet.gluon import nn, HybridBlock


class BidirectionalLSTM(HybridBlock):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = mx.gluon.rnn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Dense(nOut)

    def hybrid_forward(self, F, x, *args, **kwargs):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.shape()
        t_rec = recurrent.reshape((T * b, h))
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.reshape((T, b, -1))
        return output





if __name__ == '__main__':
    import torch

    device = torch.device('cuda:0')
    net = CRnn(32, 3, 11, 256, lstmFlag=True).to(device)
    a = torch.randn(2, 3, 32, 320).to(device)
    print(net)
    import time

    # torch.save(net.state_dict(),'crnn_lite.pth')
    tic = time.time()
    for i in range(100):
        b = net(a)
    print(b.shape)
    print((time.time() - tic) / 100)
