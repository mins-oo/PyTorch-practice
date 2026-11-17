# Model

```python
import torch
import torchvision.models as models
```

## 모델 저장

학습한 매개변수를 `state_dict`라고 불리는 내부 상태 사전(internal state dictionary)에 저장

`torch.save` 메소드를 사용하여 저장

```python
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth') # 모델 구조 자체를 저장하려면 model을 인자로 전달
```

## 모델 로드

동일 모델의 인스턴스를 생성한 다음에 `load_state_dict()` 메소드를 사용하여 매개변수들을 로드

```python
model = models.vgg16() # weights 를 지정하지 않았으므로, 학습되지 않은 모델을 생성
model.load_state_dict(torch.load('model_weights.pth'))  # 모델 구조를 로드하려면 model = torch.load('model.pth')
```

