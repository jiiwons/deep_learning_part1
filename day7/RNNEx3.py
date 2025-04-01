import torch
import torch.optim as optim
import numpy as np

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
print(sentence)
char_set = list(set(sentence))
print(char_set)
char_dic = {c:i for i,c in enumerate(char_set)}
print(char_dic)

dic_size = len(char_dic)

hidden_size = dic_size
sequence_length = 10
learning_rate = 0.1

x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length] # 입력 문자열 (10개 문자)
    y_str = sentence[i+1 : i+sequence_length+1] # 출력 문자열(입력보다 한글자 뒤) 
    print(i, x_str, '->', y_str)

    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])

print(x_data[0])
print(y_data[0])

print(np.eye(10)[1])
# 원핫 인코딩 변환 
x_one_hot = [np.eye(dic_size)[x] for x in x_data] # 문자 인덱스를 원핫 벡터로 변환
print(x_one_hot)
x = torch.FloatTensor(x_one_hot) #(배치 크기, 시퀀스 길이, 입력 크기)
y = torch.LongTensor(y_data) #(배치 크기, 시퀀스 길이)

import torch.nn as nn

class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, layers):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers = layers, batch_first=True) # batch_first=True -> (batch, seq_len, input_size) 형태의 입력을 받음
        self.fc =  nn.Linear(hidden_size, hidden_size) # RNN의 출력값을 최종 문자 예측을 위한 선형 변환

    def forward(self, x):
        output, hidden_state = self.rnn(x)
        y = self.fc(output)
        return y

model = RNNNet(dic_size, hidden_size, layers=2)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

outputs = model(x)
print(outputs.shape) # torch.Size([170, 10, 25])
print(outputs.view(-1, dic_size).shape) # torch.Size([1700, 25])

print(y.shape) # torch.Size([170, 10])
print(y.view(-1).shape)

for epoch in range(100):
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = loss_func(hypothesis.view(-1, dic_size), y.view(-1))
    loss.backward()
    optimizer.step()

    prediction = hypothesis.argmax(dim=2) # 이거 왜 2인지 설명 추가하기
                                        # hypothesis.shape = (batch_size, sequence_length, dic_size)
                                        # argmax(dim=2) : dic_size차원에서 최댓값 인덱스를 뽑음
                                        #               : (batch_size, sequence_length) 형태가 됨
    predict_str = ''
    for j, result in enumerate(prediction):
        if j == 0:
            predict_str += ''.join([char_set[t] for t in result]) # 첫번째 배치 전체 
        else:
            predict_str += char_set[result[-1]] # 이후 배치에서 마지막 문자만 추가

    print(predict_str)


