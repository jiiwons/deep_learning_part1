import torch

w = torch.tensor(2., requires_grad=True)
y = 7 * w
y.backward() # y를 미분해줌
print('w로 미분한 값:', w.grad)
print()

w2 = torch.tensor(3., requires_grad=True)
for epoch in range(20):
    y2 = 5 * w2
    y2.backward()
    print('w2로 미분한 값:', w2.grad)
    # 미분되어있는 값을 누적된 형태로 사용하도록 되어있음. 그래서 아래철머 grad.zero_를 사용해줘야함
    w2.grad.zero_()
