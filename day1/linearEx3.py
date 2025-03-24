import torch

x_train = torch.FloatTensor([[73,80,75],
                            [93,88,92],
                            [89,91,90],
                            [96,98,100],
                            [73,65,70]])

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

import torch.optim as optim

# 옵티마이저 설정
optimizer = optim.SGD([W, b], lr=1e-5)

for epoch in range(1000):
    hypothesis = x_train.matmul(W) + b  # 예측값
    loss = torch.mean((hypothesis - y_train) ** 2)  # MSE 손실함수

    optimizer.zero_grad()  # 기울기 초기화
    loss.backward()  # 기울기 계산(역전파)
    optimizer.step()  # 가중치 업데이트

    if epoch % 100 == 0:  # 100번마다 학습상태 출력
        print(
            f'epoch:{epoch + 1}, w1:{W[0].item():.3f}, w2:{W[1].item():.3f}, w3:{W[2].item():.3f}')