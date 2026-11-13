# Transform

머신러닝 학습을 위해 데이터를 적합하게 변형 <!-- preprocessing? -->

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```
FashionMNIST의 feature은 PIL Image, label은 integer다.

학습을 하려면 feature는 `FloatTensor`, label은 원-핫(one-hot)으로 encode된 tensor가 필요

이러한 변형을 위해 `ToTensor`[^1] 와 `Lambda`[^2] 를 사용

[^1]: 만능 caster!
[^2]: 코드는 사용자 정의 Lamda 함수