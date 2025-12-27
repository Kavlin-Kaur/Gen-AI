import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = 3.5
print("Probability:", sigmoid(z))
