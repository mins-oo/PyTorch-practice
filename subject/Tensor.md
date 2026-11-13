# Tensor
```python
data = [[1, 1, 1], [0, 0, 1], [1, 1, 0]]
tensor = torch.tensor(data)

Output:
tensor([[1, 1, 1],
        [0, 0, 1],
        [1, 1, 0]])
```
`tensor.shape`, `tensor.dtype`, `tensor.device`

row[i]: `tensor[i]`, column[-1]: `tensor[:, -1]`

## copy override
maintain shape, data type
```python
ones_like(tensor)
rand_like(tensor, dtype=torch.float)
zeros_like(tensor)
```

## matrix operations
```python
torch.cat((tensor, tensor), dim=1)  # dim=0: row, dim=1: column

tensor @ tensor.T  # matrix multiplication: tensor.matmul(tensor2.T)

tensor * tensor  # element-wise product: tensor.mul(tensor2)
```

tensor`s change affects numpy array
```python
n = tensor.numpy()
tensor.add_(1)

Output:
[[2 2 2]
 [1 1 2]
 [2 2 1]]
```