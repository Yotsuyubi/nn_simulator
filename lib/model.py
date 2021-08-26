import torch as th
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 256, 5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(256, 512, 5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(8208, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64*35),
            nn.ReLU(),
        )

        def out_factory(index):
            layers = [
                nn.ConvTranspose1d(64, 32, 15, stride=2, padding=7),
                nn.ReLU(),
                nn.ConvTranspose1d(32, 16, 15, stride=2, padding=7),
                nn.ReLU(),
                nn.ConvTranspose1d(16, 8, 15, stride=2, padding=7),
                nn.ReLU(),
                nn.Conv1d(8, 1, 24),
            ]
            if index <= 5:
                # layers.append(nn.ReLU())?
                pass
            return nn.Sequential(*layers)

        self.out = nn.ModuleList(
            [
                out_factory(index)
                for index in range(8)
            ]
        )

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
