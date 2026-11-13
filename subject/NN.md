# Neural Network

신경망은 데이터에 대한 연산을 수행하는 layer/module로 구성된 **module**[^1]

PyTorch의 모든 모듈은 `nn.Module` 의 하위 클래스
- `nn.Identity()`: 변화 없이 그대로 통과시키는 계층 (pass)
- `nn.Flatten()`: 2차원(또는 그 이상)의 이미지 데이터를 1차원 형태의 연속된 배열로 변환
- **`nn.Linear(in_features, out_features)`**: 입력 데이터에 가중치($weight$)와 편향($bias$)을 곱하고 더하는 선형 변환을 수행
- `nn.ReLU()`: 입력값이 0보다 작으면 0으로 만들고, 0보다 크면 그대로 통과시키는 활성화 함수(비선형)
- `nn.Sigmoid()`: 값을 각각 [0, 1] 사이의 범위로 압축해 주는 활성화 함수(비선형)
- `nn.Softmax(dim)`: 모델이 출력한 원시 값(logits)을 합이 1이 되는 확률 값으로 변환
- `nn.Sequential(*args)`: 여러 계층들을 순서대로 묶어주는 컨테이너

[^1]: 중첩된 구조가 가능해 복잡한 아키텍처를 쉽게 구축하고 관리할 수 있다

## 사용 가능한 장치 확인
```python
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```

## 신경망 모델 정의
```python
class NeuralNetwork(nn.Module):     # nn.Module의 하위클래스로 정의
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```
필수 method: `forward`로 입력에 대한 연산을 구현. **직접 호출하는게 아니다.**
```python
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
```
입력값과 함께 instance를 호출하면 tensor를 반환

### 예제
```python
import torch
from torch import nn

input_image = torch.rand(3,28,28)

flatten = nn.Flatten()
flat_image = flatten(input_image)

layer1 = nn.Linear(in_features=28*28, out_features=20)  # linear transformation R^(28*28) -> R^20
hidden1 = layer1(flat_image)

hidden1 = nn.ReLU()(hidden1)    # omitting negative value
```
```python
import torch
from torch import nn

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```
`hidden1`과 `logits`은 같다