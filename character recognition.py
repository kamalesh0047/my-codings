import tensorflow as tf
import numpy as np # Import numpy
from tensorflow.keras.models import Sequential # Corrected import
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D # Corrected import
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt # Import matplotlib

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2))) # Corrected MaxPooling layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) # Added another Conv2D layer based on the original intent
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

predictions = model.predict(x_test[:7])
predicted_classes = np.argmax(predictions, axis=1) # Get predicted classes
print("\nPredictions:")
print(predicted_classes)

for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Actual: {y_test[i]}, Predicted: {predicted_classes[i]}') # Use predicted_classes
    plt.axis('off')
    plt.show()
