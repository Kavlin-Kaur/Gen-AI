import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

X = np.random.randn(100, 10, 1)  # 100 samples, seq_len=10
y = np.random.randn(100, 1)

model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(10,1)),
    LSTM(16),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, verbose=1)

print("Prediction:", model.predict(X[:1]))
