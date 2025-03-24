import torch

x_train = torch.FloatTensor([[73,80,75],
                            [93,88,92],
                            [89,91,90],
                            [96,98,100],
                            [73,65,70]])

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

import torch.nn as nn
import torch.optim as optim

model = nn.Linear(3,1) # Linear함수가 해주는 역할이 w1x1+w2x2+w3x3 + b 를 해줌 (입력값3, 출력값1)
optimizer = optim.SGD(model.parameters(), lr=1e-5)
loss_func = nn.MSELoss()

for epoch in range(1000):
    hypothesis = model(x_train)
    loss = loss_func(hypothesis,y_train)
    optimizer.zero_grad()  # 기울기 초기화
    loss.backward()  # 기울기 계산(역전파)
    optimizer.step()  # 가중치 업데이트

    if epoch % 100 == 0:  # 100번마다 학습상태 출력
        # print(f'epoch:{epoch + 1}, parameters:{model.parameters()}') # parameters:<generator object Module.parameters at 0x00000284D5B9DC10> 제너레이터 객체 안에 파라미터가 들어가있음
        print(f'epoch:{epoch+1}',end='')
        for param in model.parameters():
            print(param, end=' ') # 첫 번째 텐서는 W (가중치)이고, 두 번째 텐서는 b (편향) 입니다.
                                # 학습이 진행될수록 W와 b 값이 점점 최적화됩니다.
        print()