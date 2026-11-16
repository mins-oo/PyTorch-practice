# Optimization

## Training Progress

1. 모델이 출력을 추측

2. 추측과 정답 사이의 오차(loss)를 계산

3. 매개변수에 대한 오차의 도함수를 [수집](/subject/Autograd.md)

4. 경사하강법으로 매개변수를 최적화

## 기본 코드

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets(...train=True, transform=ToTensor())
test_data = datasets(...train=False, ...)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(...)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
```
대략적인 베이스 코드 (데이터셋, 신경망 인스턴스)

## Hyperparameter

모델 최적화 과정을 제어할 수 있는 조절 가능한 매개변수

- 에폭(epoch): 데이터셋을 반복하는 횟수

    - 학습 단계(train loop): 학습용 데이터셋을 반복해 최적의 매개변수로 수렴

    - 검증/테스트 단계(validation/test loop): 모델 성능이 개선되고 있는지를 확인하기 위해 테스트 데이터셋을 반복

- 배치 크기(batch size): 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플의 수

- 학습률(learning rate): 각 배치/에폭에서 모델의 매개변수를 조절하는 비율. 값이 작을수록 학습 속도가 느려지고, 값이 크면 구간 벗어날 수 있다.

$$\text{1 Epoch} = \text{Iterations} \times \text{Batch Size} \approx \text{전체 데이터 개수}$$

## Loss Function

획득한 결과와 실제 값(label) 사이의 틀린 정도(degree of dissimilarity)를 측정

General Loss Function
- `nn.MSELoss`: 평균 제곱 오차 (Mean Square Error)

- `nn.NLLLoss`: 음의 로그 우도 (Negative Log Likelihood)

- `nn.CrossEntropyLoss`: `nn.LogSoftmax`와 `nn.NLLLoss`를 합침

다음 과정(optimizer)에서 이를 통해 구한 오차(loss)를 최소화할 것

## Optimizer

**확률적 경사하강법(SGD; Stochastic Gradient Descent)**, ADAM, RMSProp 등의 최적화 알고리즘 사용

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

```python
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()   # 모델을 학습(train) 모드로 설정
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)         # 예측
        loss = loss_fn(pred, y) # 오차

        loss.backward() # 역전파
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * batch_size + len(X)

def test_loop(dataloader, model, loss_fn):
    model.eval()    # 모델을 평가(eval) 모드로 설정
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # torch.no_grad()를 사용하여 평가중 변화도(gradient)를 계산하지 않도록 함
    with torch.no_grad():   
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

# ---

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
```