import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import cv2


# Function to preprocess images
def preprocess_image(img):
    img = cv2.cvtColor(
        img, cv2.COLOR_BGR2RGB
    )  # Convert BGR to RGB (assuming OpenCV reads images in BGR format)
    img = cv2.resize(img, (224, 224))  # Resize image
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Gaussian blur for noise reduction
    img = img / 255.0  # Normalize pixel values to range [0, 1]

    return img


def train_data_generator(batch_size=32):
    train_folder = os.path.join("data", "melanoma_cancer_dataset", "train")
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)

    train_generator = train_datagen.flow_from_directory(
        train_folder, target_size=(224, 224), batch_size=batch_size, class_mode="binary"
    )

    return train_generator

# Function to load image data and labels for testing
def load_test_image_data():
    test_image_data = []
    test_labels = []

    test_folder = os.path.join('data', 'melanoma_cancer_dataset', 'test')

    for label in ['malignant', 'benign']:
        label_folder = os.path.join(test_folder, label)
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            img = cv2.imread(img_path)  # Assuming you are using OpenCV for image processing
            img = preprocess_image(img)  # Apply preprocessing
            test_image_data.append(img)
            test_labels.append(label)

    # test_image_data = np.array(test_image_data)
    test_labels = np.array(test_labels)

    return test_image_data, test_labels


# Function to train and evaluate the cancer detection model using CNNs
def train_cancer_detection_model(batch_size=32):
    train_generator = train_data_generator(batch_size=batch_size)

    # Define CNN architecture
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(224, 224, 3)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(train_generator, epochs=10)
    return model


# Call the function to train the cancer detection model
cancer_model = train_cancer_detection_model(batch_size=32)


# Function to encode labels
def encode_labels(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels

# Load test data
X_test, y_test = load_test_image_data()
# Encode test labels
y_test_encoded = encode_labels(y_test)

# Evaluate the model
loss, accuracy = cancer_model.evaluate(np.array(X_test), y_test_encoded)
print("Test Accuracy:", accuracy)


# Save the trained model
cancer_model.save('cancer_detection_model.keras')


# Load the trained models
cancer_model = load_model('cancer_detection_model.keras')

# Preprocess the input image for cancer detection
def preprocess_cancer_input_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to read the image file")
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Predict the class of the input image for cancer detection
def predict_cancer_class(image_path):
    try:
        input_image = preprocess_cancer_input_image(image_path, target_size=(224, 224))
        predictions = cancer_model.predict(input_image)
        if predictions[0][0] > 0.5:
            return "Malignant"
        else:
            return "Benign"
    except Exception as e:
        print("Error:", e)
        return "Unknown"
