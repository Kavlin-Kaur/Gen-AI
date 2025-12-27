import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding

X = np.random.randint(0, 500, (200, 20))  # 200 samples of length 20
y = np.random.randint(0, 2, (200,))

model = Sequential([
    Embedding(500, 32, input_length=20),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=3, verbose=1)

print("Prediction:", model.predict(X[:1]))
