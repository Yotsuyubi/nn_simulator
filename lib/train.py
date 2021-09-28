import yaml
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import numpy as np
from .model import Model, Fresnel
from .dataset import NNSimDataset
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import argparse

pl.seed_everything(0)
th.manual_seed(0)


class LitNNSimulator(pl.LightningModule):
    def __init__(
        self, freq_list_txt, out_shape=250,
        learning_rate=1e-3, a=0.5e-6,
        resolution=0.0625, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters("out_shape")
        self.save_hyperparameters("learning_rate")
        self.save_hyperparameters("a")
        self.save_hyperparameters("resolution")

        self.model = Model(out_shape=self.hparams.out_shape)
        self.freq = np.loadtxt(freq_list_txt, delimiter=',')
        assert len(self.freq) == self.hparams.out_shape
        self.fresnel = Fresnel(self.freq)

    def forward(self, *x):
        return self.model(*x)

    def spectrum(self, *x):
        thickness = (
            (th.max(x[1], dim=1)[1]+1) *
            self.hparams.a / self.hparams.resolution
        ).unsqueeze(1).unsqueeze(1).repeat(1, 2, 1)
        return self.fresnel(self(*x), thickness)

    def configure_optimizers(self):
        optimizer = th.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate
        )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.spectrum(*x)
        l1_loss = F.l1_loss(y_hat, y)
        self.log('loss', l1_loss, prog_bar=True)
        return l1_loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.spectrum(*x)
        loss = F.l1_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train nn-simulator')
    parser.add_argument('--hparams', default="hparams.yml")
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    with open(args.hparams, 'r') as yml:
        config = yaml.safe_load(yml)

    test_split_rate = config["test_rate"]
    valid_split_rate = config["valid_rate"] + test_split_rate

    # data
    dataset = NNSimDataset(root="dataset")
    train_num = int(len(dataset)*(1.0 - valid_split_rate))
    train, val = random_split(
        dataset, [train_num, len(dataset)-train_num],
        generator=th.Generator().manual_seed(0)
    )
    test_num = int(len(val)*test_split_rate / valid_split_rate)
    val, test = random_split(
        val, [len(val)-test_num, test_num],
        generator=th.Generator().manual_seed(0)
    )

    batchsize = config["batchsize"]
    train_loader = DataLoader(train, batch_size=batchsize,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val, batch_size=batchsize,
                            shuffle=False, num_workers=4)
    test.dataset.dataset.test()
    test_loader = DataLoader(test, batch_size=batchsize,
                             shuffle=False, num_workers=4)

    # model
    model = LitNNSimulator(freq_list_txt="freq.txt",
                           learning_rate=config["learning_rate"])

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/', save_last=True,
        monitor="loss"
    )

    # training
    gpu = 0
    if th.cuda.is_available():
        gpu = 1
    trainer = pl.Trainer(
        gpus=gpu, callbacks=[checkpoint_callback],
        resume_from_checkpoint=args.resume,
        max_epochs=999
    )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(test_loader)
