from .train import LitNNSimulator
from torchvision.io import read_image
import torch as th
import numpy as np
import matplotlib.pyplot as plt

model = LitNNSimulator.load_from_checkpoint(
    "checkpoints/last.ckpt", freq_list_txt="freq.txt"
)
model.eval()

image = read_image("dataset/imgs/0_6.png").unsqueeze(0)/255
thickness = th.eye(16)[5].float().unsqueeze(0)

y = model.spectrum(image, thickness).detach().numpy()[0]

S11_0 = y[0] + 1j*y[1]
S11_90 = y[2] + 1j*y[3]
S21_0 = y[4] + 1j*y[5]
S21_90 = y[6] + 1j*y[7]

plt.figure()
plt.plot(model.freq, np.abs(S21_0) ** 2, 'r-', label='tran0')
plt.plot(model.freq, np.abs(S11_0) ** 2, 'b-', label='refl0')
plt.plot(model.freq, np.abs(S21_90) ** 2, 'r--', label='tran90')
plt.plot(model.freq, np.abs(S11_90) ** 2, 'b--', label='refl90')
plt.xlabel("Freq (THz)")
plt.ylabel("Transmittance/Reflectance (-)")
plt.legend()
plt.savefig('amp.png')
plt.close()

plt.figure()
plt.plot(model.freq, np.angle(S21_0), 'r-', label='tran0')
plt.plot(model.freq, np.angle(S11_0), 'b-', label='refl0')
plt.plot(model.freq, np.angle(S21_90), 'r--', label='tran90')
plt.plot(model.freq, np.angle(S11_90), 'b--', label='refl90')
plt.xlabel("Freq (THz)")
plt.ylabel("Transmittance/Reflectance (-)")
plt.legend()
plt.savefig('angle.png')
plt.close()
