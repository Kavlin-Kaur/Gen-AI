import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, TimeDistributed

X = np.random.randn(50, 5, 1)  
y = np.random.randn(50, 5, 1)  

model = Sequential([
    SimpleRNN(32, return_sequences=True, input_shape=(5,1)),
    TimeDistributed(Dense(1))
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, verbose=1)

print("Prediction shape:", model.predict(X[:1]).shape)
