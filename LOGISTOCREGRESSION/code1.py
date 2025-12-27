import numpy as np

# Input x and actual labels y
x = np.array([0, 1, 2, 3])
y = np.array([0, 0, 1, 1])

# Parameters
w, b = 0.0, 0.0
lr = 0.1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for epoch in range(1000):
    z = w * x + b
    y_pred = sigmoid(z)

    # gradients
    dw = np.mean((y_pred - y) * x)
    db = np.mean(y_pred - y)

    # update
    w -= lr * dw
    b -= lr * db

print("Weight:", w)
print("Bias:", b)
