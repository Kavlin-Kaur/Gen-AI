import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding

vocab_size = 5
X = np.random.randint(0, vocab_size, (100, 5))
y = np.random.randint(0, vocab_size, (100,))

model = Sequential([
    Embedding(vocab_size, 8, input_length=5),
    SimpleRNN(16),
    Dense(vocab_size, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=3, verbose=1)

print("Prediction:", model.predict(np.array([[0,1,2,3,4]])))
