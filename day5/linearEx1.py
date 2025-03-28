import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

sdata = np.array([
    [166, 59],
    [176, 75.7],
    [171, 62.1],
    [173, 70.4],
    [169, 60.1]
])

x = sdata[:, 0]
y = sdata[:, 1]

# plt.scatter(x, y, s=50)
# plt.xlabel('신장(cm)')
# plt.ylabel('몸무게(kg)')
# plt.title('신장과 체중의 관계')
# plt.show()

# 스케일
x = x - x.mean()
y = y - y.mean()

# plt.scatter(x, y, s=50)
# plt.xlabel('신장(cm)')
# plt.ylabel('몸무게(kg)')
# plt.title('신장과 체중의 관계')
# plt.show()

import torch

x_data = torch.tensor(x).float()
y_data = torch.tensor(y).float()

w = torch.tensor(1., requires_grad=True).float() # requires_grad가 활성화되어야 자동 미분 가능
b = torch.tensor(1., requires_grad=True).float()

def predict(x):
    return w * x + b

hypothesis = predict(x_data)
print(hypothesis)

def mse(h, y):
    loss = ((h-y)**2).mean()
    return loss

lr = 0.001
history = np.zeros((0,2))

for epoch in range(500):
    hypothesis = predict(x_data)
    loss = mse(hypothesis, y_data)

    loss.backward()

    lr=0.001
    # 여기가 곧 optimizer 방식 - 최소값을 찾아가는 것 최적화 = Optimization
    with torch.no_grad():
        # 경사 하강법을 이용한 w,b 값 업데이트(옵티마이저의 역할을 수행하는 부분)
        w -= lr * w.grad # 가중치 업데이트 # 에러 나는 이유(RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.): 이미 미분해버렸는데 미분을 해버리니까 미분 추적 안되도록 with torch.no_grad()를 사용해야 함
        b -= lr * b.grad # 편향 업데이트
        w.grad.zero_() # 기울기 초기화
        b.grad.zero_()

    if epoch % 10 == 0:
        item = np.array([epoch, loss.item()])
        history = np.vstack((history, item))
        print(f'epoc:{epoch+1}, loss:{loss.item():.4f}')

plt.plot(history[:,0], history[:,1], 'b')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 그래프(손실)')
plt.show()