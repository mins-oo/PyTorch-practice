# Autograd

학습에는 **역전파**가 사용됨

매개변수($weight, bias$)는 손실 함수의 **변화(gradient)**에 따라 조정됨

모든 그래프에 대한 변화도를 계산하기 위해 `torch.autograd`이라는 자동 미분 엔진을 사용

```python
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)   # 매개변수 weight
b = torch.randn(3, requires_grad=True)      # 매개변수 bias
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

`requires_grad` 옵션으로 연산 과정을 기록[^1][^2] (미분 대상임을 표시할 뿐이다)

나중에 변경도 가능
```python
w.requires_grad_(True)
```

[^1]: **`Function`** 객체로 tensor의 모든 연산(순전파, 역전파)을 관리. `.grad_fn`으로 역전파 함수의 참조를 호출
[^2]: 사용하지 않는다면(False) 고정된 매개변수(frozen parameter)를 의미

## Gradient 계산

신경망에서 매개변수의 가중치를 최적화하려면 매개변수에 대한 손실함수의 도함수(derivative)를 계산해야 한다.

즉 $\frac{\partial loss}{\partial w}, \frac{\partial loss}{\partial b}$를 구해야 한다.

```python
loss.backward() # loss 함수를 각 매개변수로 미분
print(w.grad)
print(b.grad)
```
`.backward()` 손실 함수를 만든 원인들을 거꾸로 올라가며 각각이 얼마나 영향을 미쳤는지(미분값) 추적

