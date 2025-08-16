import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.optimizers import Adam
from google.colab import userdata # Import userdata to access secrets

# Install the Kaggle API client
!pip install kaggle

# Create a directory for Kaggle configuration
!mkdir -p ~/.kaggle

# Access credentials from Colab Secrets
import json
kaggle_creds = {
    'username': userdata.get('vijayluna'),
    'key': userdata.get('vijayluna')
}

# Write the credentials to kaggle.json
with open('/root/.kaggle/kaggle.json', 'w') as f:
    json.dump(kaggle_creds, f)

# Download the dataset
!kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset

# Unzip the dataset
# Check if the zip file exists before attempting to unzip
if os.path.exists('brain-tumor-mri-dataset.zip'):
    !unzip -o brain-tumor-mri-dataset.zip -d /content/brain-tumor-mri-dataset
else:
    print("Dataset zip file not found.")

# Update the dataset directory path
dataset_dir = '/content/brain-tumor-mri-dataset'
train_dir = os.path.join(dataset_dir, 'Training')
test_dir = os.path.join(dataset_dir, 'Testing')

# Check if the directories exist after unzipping
if os.path.exists(train_dir) and os.path.exists(test_dir):
    print("Training classes:", os.listdir(train_dir))
    print("Testing classes:", os.listdir(test_dir))

    # ===== 2. Data Augmentation =====
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical'
    )
    test_set = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical'
    )

    # Automatically detect number of classes
    num_classes = len(train_set.class_indices)
    class_labels = list(train_set.class_indices.keys())
    print("Detected Classes:", class_labels)

    # ===== 3. Model (ResNet101 as Feature Extractor) =====
    # Note: ResNet101 expects input shape (224, 224, 3), but the data generators are set to (128, 128).
    # We need to either change the target_size in ImageDataGenerator or adjust the base_model input shape.
    # Let's change the target_size in ImageDataGenerator to match ResNet101 input.
    train_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224), # Changed target_size
        batch_size=16,
        class_mode='categorical'
    )
    test_set = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224), # Changed target_size
        batch_size=16,
        class_mode='categorical'
    )


    base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base model

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=CategoricalCrossentropy(),
                  metrics=['accuracy'])

    # ===== 4. Training =====
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        train_set,
        validation_data=test_set,
        epochs=20,
        callbacks=[early_stop]
    )

    # ===== 5. Save Model =====
    model_path = "/content/brain_tumor_model.h5" # Changed save path to Colab environment
    model.save(model_path)
    print(f"Model saved at: {model_path}")

    # ===== 6. Plot Training Results =====
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # ===== 7. Test a Single Image Prediction =====
    def predict_single_image(image_path, model_path, class_labels):
        model = load_model(model_path)

        # Image needs to be resized to match the input shape the model was trained on (224, 224)
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100

        print(f"Predicted: {class_labels[predicted_class]} ({confidence:.2f}%)")
        plt.imshow(load_img(image_path))
        plt.title(f"Prediction: {class_labels[predicted_class]} ({confidence:.2f}%)")
        plt.axis('off')
        plt.show()

    # Example: pick a sample image from the first detected class
    sample_image_path = os.path.join(test_dir, class_labels[0], os.listdir(os.path.join(test_dir, class_labels[0]))[0])
    predict_single_image(sample_image_path, model_path, class_labels)


else:
    print("Dataset directories not found after unzipping. Please check the unzipping process.")
