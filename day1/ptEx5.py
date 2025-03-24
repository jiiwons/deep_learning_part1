import torch

t1 = torch.tensor([1,2,3,4,5,6]).view(3,2)
t2 = torch.tensor([7,8,9,10,11,12]).view(2,3)
print(t1)
print()
print(t2)
print()

t3 = torch.mm(t1,t2) # mm:행렬과 행렬의 곱(내적)
print(t3) # 3 x 3 크기
print()

t4 = torch.matmul(t1,t2) # mm이랑 동일한 결과
print(t4)

