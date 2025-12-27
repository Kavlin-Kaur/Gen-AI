import numpy as np

m, c = 0, 0
lr = 0.01

x = np.array([1,2,3,4])
y = np.array([3,6,9,12])

for _ in range(1000):
    y_pred = m*x + c
    dm = -2 * (x*(y - y_pred)).mean()
    dc = -2 * (y - y_pred).mean()
    m -= lr*dm
    c -= lr*dc

print(m, c)
