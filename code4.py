import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.GRU(2, input_shape=(3,2))  # 3 time steps, 2 features
])
x = tf.constant([[[0.5, 0.1], [0.2, 0.3], [0.1, 0.4]]])
output = model(x)
print("GRU output:", output)
