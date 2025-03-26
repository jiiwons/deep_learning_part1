import numpy as np

wsum = np.array([0.9, 2.9, 4.0])

def softmax(ws):
    exp_a = np.exp(ws)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# 오버플로우 방지 - 지수함수를 그대로 사용하면 큰 값이 들어올 때 오버플로우 발생할 수 있으므로 최댓값을 빼는 방법을 사용하면 안정화할 수 있음 
def softmax2(ws):
    c = np.max(ws)
    exp_a = np.exp(ws - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

output = softmax(wsum)
print(output)
print(output.sum())

output2 = softmax2(wsum)
print(output2)
print(output2.sum())
