import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

train_data = datasets.FashionMNIST(
    root="data",
    train=True,             # test data는 train=False
    download=True,
    transform=ToTensor()
)


figure = plt.figure(figsize=(6, 6))

cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]     # train data라 label이 있다?
    figure.add_subplot(rows, cols, i)
    plt.title(label, fontsize=8)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

plt.show()