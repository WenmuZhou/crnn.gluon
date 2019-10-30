import numpy as np
from mxnet import init, nd
from mxnet.gluon import nn, HybridBlock


class TPS_SpatialTransformerNetwork(HybridBlock):
    """ Rectification Network of RARE, namely TPS based STN """

    def __init__(self, F, I_size, I_r_size, img_channel=1):
        """ Based on RARE TPS
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            I_size : (height, width) of the input image I
            I_r_size : (height, width) of the rectified image I_r
            I_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size  # = (I_r_height, I_r_width)
        self.img_channel = img_channel
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.img_channel)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def hybrid_forward(self, F, batch_I, *args, **kwargs):
        batch_C_prime = self.LocalizationNetwork(batch_I)  # batch_size x K x 2
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)  # batch_size x n (= I_r_width x I_r_height) x 2
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
        batch_I_r = F.grid_sample(batch_I, build_P_prime_reshape, padding_mode='border')

        return batch_I_r


class MyInit(init.Initializer):
    def __init__(self, F):
        super().__init__()
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        self.initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float).reshape(-1)

    def _init_bias(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.from_numpy(self.initial_bias)


class LocalizationNetwork(HybridBlock):
    """ Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height) """

    def __init__(self, F, img_channel):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.img_channel = img_channel
        self.conv = nn.HybridSequential()
        with self.conv.name_scope():
            self.conv.add(
                nn.Conv2D(channels=64, kernel_size=3, stride=1, padding=1, use_bias=False, activation="relu"),
                nn.BatchNorm(),
                nn.MaxPool2D(pool_size=2, strides=2),  # batch_size x 64 x I_height/2 x I_width/2
                nn.Conv2D(channels=128, kernel_size=3, strides=1, padding=1, use_bias=False, activation="relu"),
                nn.BatchNorm(),
                nn.MaxPool2D(pool_size=2, strides=2),  # batch_size x 128 x I_height/4 x I_width/4
                nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1, use_bias=False, activation="relu"),
                nn.BatchNorm(),
                nn.MaxPool2D(pool_size=2, strides=2),  # batch_size x 256 x I_height/8 x I_width/8
                nn.Conv2D(channels=512, kernel_size=3, strides=1, padding=1, use_bias=False, activation="relu"),
                nn.BatchNorm(),
                nn.GlobalAvgPool2D(1)  # batch_size x 512
            )

        self.localization_fc1 = nn.HybridSequential()
        with self.localization_fc1.name_scope():
            self.localization_fc1.add(
                nn.Dense(256),
                nn.Activation('relu')
            )

        self.localization_fc2 = nn.Dense(self.F * 2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.initialize(init=init.Constant(0))
        """ see RARE paper Fig. 6 (a) """
        # ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        # ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        # ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        # ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        # ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        # initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        # self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

        self.localization_fc2.bias.initialize(init=MyInit(F))

    def hybrid_forward(self, F, batch_I, *args, **kwargs):
        """
        input:     batch_I : Batch Input Image [batch_size x I_channel_num x I_height x I_width]
        output:    batch_C_prime : Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = batch_I.shape[0]
        features = self.conv(batch_I).view(batch_size, -1)
        batch_C_prime = self.localization_fc2(self.localization_fc1(features)).view(batch_size, self.F, 2)
        return batch_C_prime


class GridGenerator(HybridBlock):
    """ Grid Generator of RARE, which produces P_prime by multipling T with P """

    def __init__(self, F, I_r_size):
        """ Generate P_hat and inv_delta_C for later """
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)
        self.register_buffer("inv_delta_C", torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())  # F+3 x F+3
        self.register_buffer("P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float())  # n x F+3

    def _build_C(self, F):
        """ Return coordinates of fiducial points in I_r; C """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def _build_inv_delta_C(self, F, C):
        """ Return inv_delta_C which is needed to calculate T """
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # F+3 x F+3
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)  # 1 x F+3
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height
        P = np.stack(  # self.I_r_width x self.I_r_height x 2
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2
        )
        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)

        return P_hat  # n x F+3

    def build_P_prime(self, batch_C_prime):
        """ Generate Grid from batch_C_prime [batch_size x F x 2] """
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(
            batch_size, 3, 2, device=batch_C_prime.device, dtype=torch.float)), dim=1)  # batch_size x F+3 x 2
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2
        return batch_P_prime  # batch_size x n x 2
