import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)   # 매개변수 weight
b = torch.randn(3, requires_grad=True)      # 매개변수 bias
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(loss)

loss.backward()
print(w.grad)
print(b.grad)