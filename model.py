import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class UniversalMedicalImageClassifier:
    def __init__(self, img_size=150):
        """
        Class constructor.
        Parameters:
            img_size (int): Target square image size (width and height in pixels) 
                            to which all input images will be resized.
        """
        self.img_size = img_size
        self.labels = None   # List of class names (e.g. ['normal', 'pneumonia', 'covid'])
        self.model = None    # Built and trained Keras model instance
        self.history = None  # Training history log for metrics visualization

    def load_data(self, dataset_dir):
        """
        Loads images from subdirectories inside dataset_dir.
        
        Expected folder structure:
            dataset_dir/
                class_1/
                    img1.png
                class_2/
                    img1.png
        
        Images are loaded in grayscale and resized to img_size x img_size.
        Returns:
            x (np.ndarray): Image array of shape [samples, img_size, img_size]
            y (np.ndarray): Integer target class labels
        """
        data = []
        # Discover class folders
        self.labels = sorted(os.listdir(dataset_dir))
        print(f"Discovered classes: {self.labels}")

        for label in self.labels:
            path = os.path.join(dataset_dir, label)
            class_idx = self.labels.index(label)
            
            if not os.path.isdir(path):
                continue

            for img_file in os.listdir(path):
                try:
                    img_path = os.path.join(path, img_file)
                    # Load image in grayscale
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Warning: Could not read image {img_path}. Skipping.")
                        continue
                    
                    # Resize image
                    img_resized = cv2.resize(img, (self.img_size, self.img_size))
                    data.append([img_resized, class_idx])
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")

        # Shuffle dataset for uniform distribution
        np.random.shuffle(data)
        x = np.array([item[0] for item in data], dtype=np.float32)
        y = np.array([item[1] for item in data], dtype=np.int32)
        print(f"Total images loaded: {len(x)}")
        return x, y

    def preprocess_data(self, x, y):
        """
        Prepares raw image arrays and labels for training.
        - Pixel normalization to range [0, 1]
        - Expands shape to 4D tensor [batch, height, width, channels]
        - One-hot encodes target labels
        """
        x = x / 255.0
        x = x.reshape(-1, self.img_size, self.img_size, 1)
        y = to_categorical(y, num_classes=len(self.labels))
        return x, y

    def build_model(self):
        """
        Constructs and compiles the Convolutional Neural Network (CNN) architecture.
        """
        model = Sequential([
            # Convolutional Block 1
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(self.img_size, self.img_size, 1)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Convolutional Block 2
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.3),

            # Convolutional Block 3
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.4),

            # Fully Connected Dense Layers
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(len(self.labels), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        print(self.model.summary())

    def train(self, x_train, y_train, x_val, y_val, batch_size=32, epochs=30):
        """
        Trains the CNN model using data augmentation and dynamic learning rate callbacks.
        """
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        datagen.fit(x_train)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)

        self.history = self.model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[reduce_lr, early_stop]
        )
        return self.history

    def evaluate(self, x_test, y_test):
        """Evaluates model performance on the hold-out test dataset."""
        loss, acc = self.model.evaluate(x_test, y_test)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc * 100:.2f}%")
        return loss, acc

    def predict(self, img):
        """
        Runs inference on a single input grayscale medical image.
        Returns predicted class index and confidence score.
        """
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_norm = img_resized / 255.0
        img_input = img_norm.reshape(1, self.img_size, self.img_size, 1)
        probs = self.model.predict(img_input)[0]
        class_idx = np.argmax(probs)
        confidence = probs[class_idx]
        return class_idx, confidence

    def save_model(self, path):
        """Saves current trained model weights and architecture to disk."""
        self.model.save(path)
        print(f"Model saved successfully to: {path}")

    def load_model(self, path):
        """Loads pre-trained model weights from disk."""
        self.model = load_model(path)
        print(f"Model loaded successfully from: {path}")


if __name__ == "__main__":
    classifier = UniversalMedicalImageClassifier(img_size=150)
    print("UniversalMedicalImageClassifier initialized successfully.")
