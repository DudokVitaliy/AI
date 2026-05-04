import torch
t = torch.tensor([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])

print("Reshaping")
print(t.reshape(6, 2))

print("\nResizing")
print(t.view(2, 6))

print("\nTransposing")
print(t.transpose(0, 1))

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)