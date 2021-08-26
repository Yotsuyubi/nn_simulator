import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import numpy as np
from .model import Model
from .dataset import NNSimDataset


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
        k_0 = 2.0 * np.pi * self.freq.repeat(x.shape[0], 2, 1) / self.C
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


class LitNNSimulator(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.fresnel = Fresnel(th.linspace(0.5e12, 1e12, steps=250))

    def forward(self, *x):
        return self.model(*x)

    def spectrum(self, *x):
        thickness = (
            (th.max(x[1], dim=1)[1]+1) * 0.5e-6 / 0.0625
        ).unsqueeze(1).unsqueeze(1).repeat(1, 2, 1)
        return self.fresnel(self(*x), thickness)

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.spectrum(*x)
        l1_loss = F.l1_loss(y_hat, y)
        self.log('train_loss', l1_loss, prog_bar=True)
        return l1_loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.spectrum(*x)
        loss = F.l1_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)


if __name__ == "__main__":
    # data
    dataset = NNSimDataset(root="dataset")
    # train, val = random_split(dataset, [7000, 1000])

    train_loader = DataLoader(dataset, batch_size=32,
                              shuffle=True, num_workers=4)
    # val_loader = DataLoader(val, batch_size=256, shuffle=True, num_workers=4)

    # model
    model = LitNNSimulator()

    # training
    trainer = pl.Trainer(gpus=0)
    trainer.fit(model, train_loader)
