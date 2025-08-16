import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
text = "language modeling using rnn is fun"
chars = sorted(set(text))  # Unique characters
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
seq = [char2idx[c] for c in text]
seq_length = 10
X = []
Y = []

for i in range(len(seq) - seq_length):
    X.append(seq[i:i+seq_length])
    Y.append(seq[i + seq_length])
X = np.array(X)
Y = np.array(Y)
vocab_size = len(chars)
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=16, input_length=seq_length),
    SimpleRNN(64),
    Dense(vocab_size, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=20, verbose=2)
def generate_text(model, seed, length):
    input_seq = [char2idx[c] for c in seed[-seq_length:]]
    for _ in range(length):
        input_one = np.array([input_seq[-seq_length:]])
        pred = model.predict(input_one, verbose=0)
        next_char_idx = np.argmax(pred)
        input_seq.append(next_char_idx)
    return ''.join([idx2char[i] for i in input_seq])

seed = "language m"
print("Generated Text:", generate_text(model, seed, 60))
