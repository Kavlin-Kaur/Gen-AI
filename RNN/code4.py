import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

X = np.random.randn(100, 10, 1)  
y = np.random.randn(100, 1)

model = Sequential([
    SimpleRNN(32, return_sequences=True, input_shape=(10,1)),
    SimpleRNN(16),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, verbose=1)

print("Prediction:", model.predict(X[:1]))
