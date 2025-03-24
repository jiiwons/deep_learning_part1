import torch

t1 = torch.linspace(0,3,10)
print(t1)
print()
print(torch.exp(t1))
print(torch.log(t1))
print(torch.cos(t1))
print(torch.sqrt(t1))
print(torch.mean(t1))
print(torch.std(t1))
print()

t2 = torch.tensor([[2,4,6], [7,3,5]])
print(t2)
print()

print(torch.max(t2))
print()

print(torch.max(t2, dim=1)) #dim 썼을 때랑 안썼을때랑 다름, 1은 axis=1을 의미 1축에서 가장 큰 값을 전달하고 인덱스 값 반환
print()

print(torch.max(t2, dim=1)[0]) # value
print(torch.max(t2, dim=1)[1]) # index