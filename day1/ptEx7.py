import torch
import torch.nn.init as init

t1 = init.uniform_(torch.FloatTensor(3,4)) # 3 x 4 의 객체에 들어가는 값은 uniform_(random variable)으로 채움
print(t1)
print()

t2 = init.normal_(torch.FloatTensor(3,4), mean=10, std=3)
print(t2)
print()

t3 = torch.FloatTensor(torch.randn(3,4)) # 표준정규분포로부터 3 x 4 객체 생성
print(t3)
print()

t4 = init.constant_(torch.FloatTensor(3,4), 100)
print(t4)

