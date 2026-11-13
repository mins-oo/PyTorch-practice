# Dataset

## torch.utils.data 데이터셋
```python
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.FashionMNIST(
    root="data",
    train=True,             # test data는 train=False
    download=True,
    transform=ToTensor()
)
```
FashionMNIST Dataset figure 출력
```python
import matplotlib.pyplot as plt

figure = plt.figure(figsize=(8, 8))

cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label, fontsize=8)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

plt.show()
```

## 커스텀 데이터셋
```python
import os
import pandas as pd

from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)        # image -> tensor 변환
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```
필수 methods: `__init__`, `__len__`, `__getitem__`

## DataLoader
모델을 학습할 때, 일반적으로 샘플들을 `미니배치(minibatch)`로 전달하고, 매 에폭(epoch)마다 데이터를 다시 섞어서 과적합을 막고, Python의 `multiprocessing`[^1][^2] 을 사용하여 데이터 검색 속도를 높인다.

이러한 과정들을 추상화한 순회 가능한 객체(iterable)

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
```

[^1]: multiprocessing: 프로세스마다 각각 독자적인 메모리 공간과 별도의 GIL을 할당해 멀티코어 병렬처리를 가능

[^2]: GIL(Global Interpreter Lock, 전역 인터프리터 락): 파이썬 스레드는 한 번에 딱 하나의 코어에서만 실행할 수 있다. 한 메모리에 여러 접근을 막기 위함