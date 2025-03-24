import torch

x1_train = torch.FloatTensor([[73],[93],[89],[96],[73]])
x2_train = torch.FloatTensor([[80],[88],[91],[98],[65]])
x3_train = torch.FloatTensor([[75],[92],[90],[100],[70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]]) # y = w1*x1 + w2*x2 + w3*x3

w1 = torch.zeros((1,1), requires_grad=True)
w2 = torch.zeros((1,1), requires_grad=True)
w3 = torch.zeros((1,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

import torch.optim as optim
# 옵티마이저 설정
optimizer = optim.SGD([w1,w2, w3, b], lr=1e-5) # 경사 하강법(Stochastic Gradient Descent) 알고리즘 사용

for epoch in range(1000):
    hypothesis = torch.mm(x1_train, w1) + torch.mm(x2_train, w2) + torch.mm(x3_train, w3) + b # 예측값
    loss = torch.mean((hypothesis - y_train) **2) # MSE 손실함수
    
    optimizer.zero_grad() # 기울기 초기화
    loss.backward() # 기울기 계산(역전파)
    optimizer.step() # 가중치 업데이트

    if epoch % 100 == 0: # 100번마다 학습상태 출력
        print(f'epoch:{epoch+1}, w1:{w1.item()}, w2:{w2.item()}, w3:{w3.item()}, b:{b.item()}, loss:{loss.item():.3f}') # 배열가져올땐 numpy, 스칼라값 가져올 땐 item


# 입력 데이터와 출력 데이터 사이의 선형 관계를 학습합니다.
# 경사 하강법을 사용하여 손실을 최소화하는 방향으로 파라미터들을 업데이트합니다.
    # 어떻게 손실을 최소화?
    # 모델은 손실 함수(MSE)를 최소화하려고 기울기(Gradient)를 계산하고,
    # 그 기울기를 반대 방향으로 파라미터들을 업데이트해서 손실을 줄여나갑니다.
    # 이 과정이 반복되면서 점점 더 예측이 정확해지게 됩니다.