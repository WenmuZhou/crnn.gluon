import mxnet as mx
from mxnet.gluon import nn, HybridBlock

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
    def __init__(self, hidden_size, num_layers):
        super(BidirectionalLSTM, self).__init__()
        with self.name_scope():
            self.rnn = mx.gluon.rnn.LSTM(hidden_size, num_layers, bidirectional=True, layout='NTC')

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.rnn(x)
        return x


class Decoder(HybridBlock):
    def __init__(self, hidden_size=256, num_layers=1):
        super(Decoder, self).__init__()
        with self.name_scope():
            self.lstm = nn.HybridSequential()
            with self.lstm.name_scope():
                self.lstm.add(BidirectionalLSTM(hidden_size, num_layers))
                self.lstm.add(BidirectionalLSTM(hidden_size, num_layers))

    def hybrid_forward(self, F, x, *args, **kwargs):

        x = self.lstm(x)
        return x