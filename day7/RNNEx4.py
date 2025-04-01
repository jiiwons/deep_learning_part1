import torch
import torch.nn as nn
import unidecode
import string
import random

total_epochs = 2000
chunk_len = 200

hidden_size = 100 # 19번째 줄에서 n_characters의 길이가 100이라서 hidden_size도 100으로
batch_size = 1
num_layer = 1
embedding = 70
learning_rate = 0.002

print(string.printable)
all_characters = string.printable
n_characters = len(all_characters)
print(n_characters) # 100

file = unidecode.unidecode(open('input.txt').read())
# print(file)
file_len = len(file)
print(file_len)

# 랜덤 문자열 선택 함수 
def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index : end_index]

# 문자를 숫자로 변환하는 함수
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return tensor

print(char_tensor('good')) # tensor([16, 24, 24, 13])

# RNN 모델(LSTM) 구현
class RNNet(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.encode = nn.Embedding(self.input_size, self.embedding_size) # 문자 -> 벡터 변환(입력 문자를 embedding_size 차원의 벡터로 변환)
        self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, num_layers) # LSTM(LSTM레이어 추가(입력 크기: embedding_size, 출력 크기: hidden_size))
        self.fc = nn.Linear(self.hidden_size, self.output_size) # 최종 출력을 문자 개수만큼 변환(LSTM의 최종 출력을 문자 개수(output_size = n_characters)로 변환)
    
    # hidden과 cel 상태를 0으로 초기화
    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, batch_size, hidden_size)
        cell = torch.zeros(self.num_layers, batch_size, hidden_size)
        return hidden, cell

    def forward(self, input, hidden, cell):
        x = self.encode(input.view(1,-1)) # 문자 인덱스를 임베딩 벡터로 변환 
        out, (hidden, cell) = self.rnn(x, (hidden, cell))  #LSTM 실행
        y = self.fc(out.view(batch_size, -1)) # 최종 출력 변환
        return y, hidden, cell


model = RNNet(input_size=n_characters,
              embedding_size=embedding,
              hidden_size=hidden_size,
              output_size=n_characters,
              num_layers=num_layer)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def random_train_set():
    chunk = random_chunk()
    input = char_tensor(chunk[:-1]) # 입력 문자
    target = char_tensor(chunk[1:]) # 정답(다음 문자)
    return input, target

def test():
    star_str = 'b'
    input = char_tensor(star_str)
    hidden, cell = model.init_hidden()

    for i in range(200):
        output, hidden, cell = model(input, hidden, cell)
        output_dist = output.data.view(-1).div(0.8).exp()
        top_i = torch.multinomial(output_dist, 1)[0] # top_index, 확률에 따라 문자 선택
        predicted_char = all_characters[top_i]
        print(predicted_char, end='')
        input = char_tensor(predicted_char)

for epoch in range(total_epochs):
    input, label = random_train_set()
    hidden, cell = model.init_hidden()
    loss = torch.tensor([0]).type(torch.FloatTensor)

    optimizer.zero_grad()
    for j in range(chunk_len - 1):
        x_train = input[j]
        y_train = label[j].unsqueeze(dim=0).type(torch.LongTensor)
        hypothesis, hidden, cell = model(x_train, hidden, cell)
        loss += loss_func(hypothesis, y_train)

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print('='*200)
        print('loss:{:.4f}'.format(loss.item()/chunk_len), end='\n\n')
        test()
        print('\n', '='*200, end='\n\n')
