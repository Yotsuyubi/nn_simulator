import torch as th
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
import json
import os.path as path

th.manual_seed(0)


class NNSimDataset(Dataset):

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.is_test = False
        with open(self.getpath("geometry_params.json")) as fp:
            self.geometry_params = json.load(fp)

    def __len__(self):
        return len(self.geometry_params)

    def __getitem__(self, index):

        image = read_image(
            self.getpath("imgs", self.geometry_params[index]["filename"])
        )/255
        if th.rand(1) < 0.3 and not self.is_test:
            image -= th.rand(image.size())*0.9
            image = image.clamp(0.0, 1.0)

        thickness = th.eye(16)[self.geometry_params[index]["thickness"]-1]
        if th.rand(1) < 0.3 and not self.is_test:
            thickness += th.rand(thickness.size())*0.1
            thickness = th.softmax(thickness, dim=-1)  # soft-label

        spectrum = np.load(self.getpath("spectrums", "{}.npy".format(index)))

        return (image, thickness), np.array(spectrum, dtype=np.float32)

    def getpath(self, *x):
        return path.join(self.root, *x)

    def test(self):
        self.test = True


if __name__ == "__main__":
    dataset = NNSimDataset("dataset")
    (img, thickness), spectrum = dataset.__getitem__(0)
    print(img.shape, thickness, spectrum.shape)
