import torch
import torch.nn as nn
import numpy as np

string = 'hello pytorch. how long can a rnn cell remember? show me your limit!'
print(len(string))
chars = 'abcdefghijklmnopqrstuvwxyz ?!.,:01'

char_list = [i for i in chars]
n_letter = len(char_list)
print(n_letter)
print(char_list)

n_hidden = 35
learning_rate = 0.01
total_epochs = 1000

def stringToOnehot(string):
    # 모든 원소가 0인 길이 n_letter짜리 배열 start와 end를 생성. 즉, [0,0,0...,0]형태의 배열을 만듦
    start = np.zeros(n_letter, dtype=int)
    end = np.zeros(n_letter, dtype=int)
    start[-2] = 1 #[0,0,0....,0,1,0] 맨 뒤에서 두번째 거 1로 표시한 이유는 chars에서 선언한 0을 start로 쓸 거라서?
    end[-1] = 1 #[0,0,0,...,0,0,1]

    # 문자열을 원 핫 벡터로 변환 
    for i in string:
        idx = char_list.index(i) # 현재 문자 인덱스 찾기
        odata = np.zeros(n_letter, dtype=int)
        odata[idx] = 1 # 해당 문자 위치를 1로 설정(idx위치의 값을 1로 바꿔 해당 문자를 원핫인코딩함)
        start = np.vstack([start, odata]) # vertical stack으로 묶음(새로운 행 추가, 기존의 start배열과 새로운 odata배열을 수직으로 쌓아서 누적)
    output = np.vstack([start,end]) # start 배열의 마지막에 end벡터를 추가하여 문자열의 끝을 표시
    return output
print(stringToOnehot('test'))

# 원핫 벡터를 문자로 변환 
# 가장 큰 값(1이 있는 위치)의 인덱스를 찾아 문자로 변환하는 함수 
def onehotToChar(onehot_d):
    onehot = torch.Tensor.numpy(onehot_d)
    return char_list[onehot.argmax()]

data = np.zeros(n_hidden, dtype=int)
data[5]=1
print(onehotToChar((torch.from_numpy(data)))) # f / 문자 집합에서 5번째 문자를 출력

class RNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # i to hidden(입력-> 은닉 상태 변환)
        self.i2o = nn.Linear(input_size + hidden_size, output_size) # i to output(입력 -> 출력 변환)
        self.ac_fn = nn.Tanh()

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=1) # 입력과 이전 은닉 상태 연결
        hidden = self.ac_fn(self.i2h(combined)) # 새로운 은닉 상태 계산 
        output = self.i2o(combined) # 출력 계산
        return output, hidden

rnn = RNNet(n_letter, n_hidden, n_letter) #(입력크기, 은닉크기, 출력크기)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

one_hot = torch.from_numpy(stringToOnehot(string)).type_as(torch.FloatTensor())
print()
print(one_hot)

for i in range(total_epochs):
    optimizer.zero_grad()
    hidden = rnn.init_hidden()
    total_loss = 0

    for j in range(one_hot.size()[0] -1):
        input = one_hot[j:j+1, :] # 현재 문자
        target = one_hot[j+1] # 다음 문자
        hypothesis, hidden =  rnn(input, hidden) # 예측
        loss = loss_func(hypothesis.view(-1), target.view(-1)) # 손실 계산 
        total_loss += loss

    total_loss.backward()
    optimizer.step()

    # 모델 테스트 
    start = torch.zeros(1, n_letter)
    start[:, -2] = 1 # 시작 심볼 '0' 원핫 벡터
    with torch.no_grad():
        hidden = rnn.init_hidden()
        # 새로운 문자열 생성 반복(94, 95)
        input = start # 입력값을 시작 심볼로 설정(input을 '0' 원핫벡터로 초기화)
        output_string = '' # 새로운 문자열을 저장할 변수
        for i in range(len(string)):
            output, hidden = rnn(input, hidden) # 문자 예측
            output_string += onehotToChar(output.data) # 문자 변환(가장 높은 확률을 가진 문자 변환)
            input = output # 다음 입력으로 사용
    print('', output_string)

# 정리 - 새로운 문자열을 생성하는 핵심 과정
# 1. 시작 심볼(0)을 입력값으로 설정
# 2. RNN을 실행하여 하나씩 문자를 예측
# 3. 예측된 출력을 다음 입력으로 사용하여 계속 문자를 생성
# 4. 모든 문자 예측이 끝나면 최종 문자열을 출력