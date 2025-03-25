import torch
import numpy as np
import  matplotlib.pyplot as plt

# 시그모이드 함수 정의
def sigmoid(x):
    return 1 / (1+np.exp(-x))

x = np.arange(-10,10,0.1)
y = sigmoid(x)

plt.plot([0,0], [1,0], ':')
plt.plot([-10,10], [0,0], ':')
plt.plot([-10,10], [1,1], ':')

# e(ax+b)중에 a에 대해
# y1 = sigmoid(0.5*x)
# y2 = sigmoid(x)
# y3 = sigmoid(2 * x)
# plt.plot(x, y1, label='0.5*x')
# plt.plot(x, y2, label='x')
# plt.plot(x, y3, label='2*x')
# plt.grid(True)
# plt.legend(loc='best')
# plt.show() # 기울기 값이 클수록 step function 에 가까워지고, 작을수록 linear 함

# e(ax+b)중에 b에 대해
y1 = sigmoid(x - 1.5)
y2 = sigmoid(x)
y3 = sigmoid(x + 1.5)
plt.plot(x, y1, label='x - 1.5')
plt.plot(x, y2, label='x')
plt.plot(x, y3, label='x + 1.5')
plt.grid(True)
plt.legend(loc='best')
plt.show() # b가 음수면 우측, 양수면 좌측 