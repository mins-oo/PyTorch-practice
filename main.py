import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

input_image = torch.rand(3,28,28)

flatten = nn.Flatten()
flat_image = flatten(input_image)

layer1 = nn.Linear(in_features=28*28, out_features=20)  # linear transformation R^(28*28) -> R^20
hidden1 = layer1(flat_image)

hidden1 = nn.ReLU()(hidden1)    # omitting negative value

softmax = nn.Softmax(dim=1)
pred_probab = softmax(hidden1)

print(pred_probab)