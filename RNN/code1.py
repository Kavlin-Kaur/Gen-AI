import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Sequence data: X=1..9 → y=2..10
X = np.array([[[i]] for i in range(1, 10)])
y = np.array([i+1 for i in range(1, 10)])

model = Sequential([
    SimpleRNN(20, activation='tanh', input_shape=(1,1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=0)

print("Prediction for 10:", model.predict(np.array([[[10]]])))
