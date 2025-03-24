import torch

t1 = torch.tensor([1,2,3])
t2 = torch.tensor([5,6,7])
print(t1)
print(t2)
print()

t3 = t1 + 30
print(t3)
print()

t4 = t1 + t2
print(t4)
print()

t5 = torch.tensor([[10,20,30], [40,50,60]])
print(t5)
print()

print(t5 + t1)
