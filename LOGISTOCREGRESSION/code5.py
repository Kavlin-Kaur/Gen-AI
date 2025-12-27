import numpy as np

x = np.linspace(-10, 10, 50)
y = 1 / (1 + np.exp(-x))

print("Sigmoid outputs:", y[:5])
