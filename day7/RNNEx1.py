import numpy as np


time_step = 10
input_size = 4
hidden_size = 8

inputs = np.random.random((time_step, input_size))
print(inputs.shape)
hidden_state_t = np.zeros((hidden_size,)) # time_step만큼 필요한 게 아니라 딱 한번 쓸만큼만
print(hidden_state_t.shape)

# RNN 출력: h_t = tanh(W_x*x_t + W_h*h_{t-1} + b) 
# x_t : 현재 입력
# W_x : 입력 x_t에 대한 가중치 행렬
# h_{t-1}: 이전 은닉 상태
# W_h : 이전 은닉 상태 h_{t-1}에 대한 가중치 행렬
wx = np.random.random((input_size, hidden_size)) # 입력 가중치: 입력 벡터 x_t를 은닉 상태로 변한하는 가중치 
                                                # 입력 x_t의 크기는 (input_size,)이므로, 행렬 곱셈 W_x*x_t를 수행하려면 W_x는 (input_size, hidden_size)이어야 함
                                                # (W_x * x_t) = (input_size, hidden_size) X (input_size, 1) = (hidden_size, 1)
                                                # 즉, hidden_size 개의 뉴런이 출력으로 나옴
                                                
wh = np.random.random((hidden_size, hidden_size)) # 은닉 상태 가중치: 이전 은닉 상태 h_{t-1}를 새로운 은닉 상태 h_t로 변환하는 가중치
                                                # 이전 은닉 상태 h_{t-1}의 크기는 (hidden_size,)이므로, 행렬 곱셈 W_h*h_{t-1}를 수행하려면 W_h는 (hidden_size, hidden_size)이어야 함
                                                # (W_h * h_{t-1}) = (hidden_size, hidden_size) X (hidden_size, 1) = (hidden_size, 1)
                                                # 즉, hidden_size 개의 뉴런이 출력으로 나옴
                                                
b = np.random.random((hidden_size,))            # 편향 벡터 : RNN의 각 뉴런에 추가되는 편향 값 
                                                # W_x*x_t와 W_h*h_{t-1}의 결과는 (hidden_size, )크기의 벡터이므로, 이에 더할 수 있도록 b도 같은 크기인 (hidden_size,)여야함
print()
print(wx.shape)
print(wh.shape)
print(b.shape)

total_hidden_state = []
# RNN의 순전파 연산 구현 
for input_t in inputs: # inputs는 (time_step, input_size)크기의 행렬이므로 각 input_t는 크기가 (input_size,)인 벡터이고, time_step번 반복됨
    output_t = np.tanh(np.dot(input_t, wx) + np.dot(hidden_state_t, wh) + b)
    total_hidden_state.append(list(output_t))
    hidden_state_t = output_t # 현재 은닉 상태 업데이트(새로운 은닉상태 output_t를 hidden_state_t로 업데이트하여 다음 타입스텝에서 사용)

print()
total_hidden_state = np.stack(total_hidden_state) # 모든 타임스텝이 끝나면 total_hidden_state는 (time_step, hidden_size)크기의 배열로 변환됨
                                                # total_hidden_state에는 각 타임스텝에서 계산된 은닉 상태들이 저장됨
print(total_hidden_state.shape)
print(total_hidden_state)