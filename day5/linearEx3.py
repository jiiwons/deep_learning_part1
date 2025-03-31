import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn

torch.manual_seed(123)

# l1 = nn.Linear(1,1)
# print(l1)
# print()
#
# for param in l1.named_parameters():
#     print(param[0])
#     print(param[1])
#     print(param[1].shape)
# print()
#
# l2 = nn.Linear(2,1) # w1x1 + w2x2 + b
# print(l2.weight) # tensor([[-0.3512,  0.2667]], requires_grad=True)
# # nn.init.constant_()를 사용하면 가중치(weight)와 편향(bias)를 원하는 값으로 초기화할 수 있음
# nn.init.constant_(l2.weight, 1.0)
# nn.init.constant_(l2.bias, 2.0)
# print()
# print(l2.weight) # tensor([[1., 1.]], requires_grad=True)
# print(l2.bias) # tensor([2.], requires_grad=True)
import pandas as pd

data_url = 'http://lib.stat.cmu.edu/datasets/boston'

df = pd.read_csv(data_url, sep='\s+', skiprows=22, header=None)
df.info()

'''
CRIM: 해당 지역의 1인당 범죄율
ZN: 25,000 평방피트(약 6950㎡) 이상 크기의 주거 지역 비율 (%)
INDUS: 비소매 상업 지역(산업, 공장 등)이 차지하는 비율 (%)
CHAS: 찰스강(Charles River) 근처 여부 (더미 변수)(강 근처: 1,강 근처 아님: 0)
NOX: 대기 중 일산화질소(NO₂) 농도 (백만 분율 단위, ppm)
RM: 주택 1가구당 평균 방 개수
AGE: 1940년 이전에 지어진 자가 주택 비율 (%)
DIS: 보스턴 주요 5개 고용 센터까지의 가중 거리
RAD: 고속도로 접근성 지수 (숫자가 높을수록 주요 고속도로와 가까움)
TAX: 재산세율 (1만 달러당 세금)
PTRATIO: 학생-교사 비율 (낮을수록 학생 1인당 교사 수가 많음)
B: 인구 중 흑인의 비율을 나타내는 지수
LSTAT: 저소득층 비율 (%)
MEDV: 해당 지역의 주택 가격 중앙값 ($1000 단위)
'''
feature_names = np.array(['CRIM', 'ZN','INDUS','CHAS','NOX','RM','AGE',
                          'DIS','RAD','TAX','PTRATIO','B','LSTAT'])

# 짝수 번째 행과 홀수 번째 행 일부(앞 2열)로 구성된 배열이 됨
x_org = np.hstack([df.values[::2, :], # 짝수 번째 행(0, 2, 4, ...)의 모든 열 선택
                  df.values[1::2, :2]]) # 홀수 번째 행(1, 3, 5, ...)의 앞 2개 열 선택
print(x_org.shape)
yt = df.values[1::2, 2] # 집값
print(yt.shape)

x = x_org[:, feature_names=='RM']
print(x)

# plt.scatter(x, yt, s=10, c='b')
# plt.xlabel('방 개수')
# plt.ylabel('집 가격')
# plt.title('방 개수와 집 가격의 산포도')
# plt.show()

# 데이터를 학습해서 linear한 선을 그리고
# weight와 bias를 1.0으로
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, output_size)

        nn.init.constant_(self.l1.weight, 1.0)
        nn.init.constant_(self.l1.bias, 1.0)

    def forward(self,x):
        return self.l1(x)

model = Net(x.shape[1], 1)
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x_data = torch.tensor(x).float()
y_data = torch.tensor(yt).float()
y_data = y_data.view((-1,1))
print(y_data.shape)

for epoch in range(50000):
    optimizer.zero_grad()
    hypothesis = model(x_data)
    loss = loss_func(hypothesis, y_data)/2 #/2함으로써 좀 더 세밀하게
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'epoch:{epoch+1} loss:{loss.item():.4f}')

x_test = np.array((x.min(), x.max())).reshape(-1,1)
x_test = torch.tensor(x_test).float()

with torch.no_grad():
    prediction  = model(x_test)

plt.scatter(x, yt, s=10, c='b')
plt.xlabel('방 개수')
plt.ylabel('집 가격')
plt.plot(x_test.data, prediction.data, c='r')
plt.title('방 개수와 집 가격의 산포도')
plt.show()