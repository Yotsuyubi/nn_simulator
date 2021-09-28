import torch as th
import torch.nn as nn
import numpy as np


class Fresnel(nn.Module):

    def __init__(self, freq):
        super().__init__()
        self.freq = freq
        self.C = 299_792_458

    def forward(self, x, d):
        n = x[:, 0:2, :]
        k = x[:, 2:4, :]
        imp_re = x[:, 4:6, :]
        imp_im = x[:, 6:8, :]
        n_hat = th.abs(n) + 1j*th.abs(k)
        k_0 = (2.0 * np.pi *
               self.freq.repeat(x.shape[0], 2, 1) / self.C).type_as(x)
        Z1 = 1
        Z2 = th.abs(imp_re) + 1j*imp_im
        Z3 = 1
        t12 = 2 * Z2 / (Z1 + Z2)
        t23 = 2 * Z3 / (Z2 + Z3)
        r12 = (Z2 + -1 * Z1) / (Z1 + Z2)
        r23 = (Z3 + -1 * Z2) / (Z2 + Z3)
        P2 = th.exp(1j * k_0 * d * n_hat)
        t123 = (t12 * t23 * P2) / (1 + r12 *
                                   r23 * P2**2) / th.exp(1j * k_0 * d)
        r123 = (r12 + r23 * P2**2) / (1 + r12 * r23 * P2**2)
        return th.cat([
            r123[:, 0].unsqueeze(1).real, r123[:, 0].unsqueeze(1).imag,
            r123[:, 1].unsqueeze(1).real, r123[:, 1].unsqueeze(1).imag,
            t123[:, 0].unsqueeze(1).real, t123[:, 0].unsqueeze(1).imag,
            t123[:, 1].unsqueeze(1).real, t123[:, 1].unsqueeze(1).imag,
        ], dim=1)


class CenterCrop(nn.Module):

    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, tensor):
        shape = tensor.shape[-1]
        if shape == self.target_shape:
            return tensor
        diff = shape - self.target_shape
        crop_start = diff // 2
        crop_end = diff - crop_start
        return tensor[:, :, crop_start:-crop_end]


class Model(nn.Module):

    def __init__(self, out_shape):
        super().__init__()
        self.out_shape = out_shape

        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, 5, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 5, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(512+16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64*35),
            nn.ReLU(),
        )

        def out_factory():
            layers = [
                nn.ConvTranspose1d(64, 32, 15, stride=2),
                nn.ReLU(),
                nn.Conv1d(32, 32, 5),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 16, 15, stride=2),
                nn.ReLU(),
                nn.Conv1d(16, 16, 5),
                nn.ReLU(),
                nn.ConvTranspose1d(16, 8, 15, stride=2),
                nn.ReLU(),
                nn.Conv1d(8, 8, 5),
                nn.ReLU(),
                nn.ConvTranspose1d(8, 4, 15, stride=2),
                nn.ReLU(),
                nn.Conv1d(4, 1, 1),
                CenterCrop(self.out_shape)
            ]
            return nn.Sequential(*layers)

        self.out = nn.ModuleList([out_factory() for _ in range(8)])

    def forward(self, x, thickness):
        feature = th.cat([self.feature(x), thickness], dim=1)
        fc = self.fc(feature).view(-1, 64, 35)
        return th.cat([out(fc) for out in self.out], dim=1)


if __name__ == "__main__":

    model = Model()
    dummy_img = th.randn(1, 1, 32, 32)
    dummy_thickness = th.eye(16)[0].unsqueeze(0).float()
    y = model(dummy_img, dummy_thickness)
    print(y.shape)
