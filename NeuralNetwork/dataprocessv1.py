import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LSTM, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate

# Variables
data_dir = 'E:/Programming/PersonalProjects/CaptchaCoin/NeuralNetwork/base_images'
img_width = 128
img_height = 64
batch_size = 64 # Do not change
learning_rate = 0.001
epochs = 500
patience_lr = 8
patience_es = 24
dropout_var = 0.5
visualize = False  # Set to True to enable visualization, False to disable

# Load the dataset
X = []
y = []

for file in os.listdir(data_dir):
    if file.endswith('.png'):
        label = file[:-4]
        if len(label) == 3 and label.isalpha() and label.isupper():
            img_path = os.path.join(data_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_width, img_height))
            X.append(img)
            y.append(label)
        else:
            print(f"Skipping {file}: Invalid label")

X = np.array(X).reshape(-1, img_height, img_width, 1) / 255.0
y = np.array(y)

# Encode labels
label_encoder = LabelEncoder()
y_flat = np.array([char for label in y for char in label])
y_encoded = label_encoder.fit_transform(y_flat)
y_one_hot = np.eye(len(label_encoder.classes_))[y_encoded]

# Verify the number of unique classes
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")
print(f"Classes: {label_encoder.classes_}")

# Reshape y_one_hot back to the original shape
y_one_hot = y_one_hot.reshape(-1, num_classes * 3)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=10,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest'
)

# Define custom loss function
def custom_loss(y_true, y_pred):
    y_true = K.reshape(y_true, (-1, 3, num_classes))
    y_pred = K.reshape(y_pred, (-1, 3, num_classes))
    loss = K.mean(categorical_crossentropy(y_true, y_pred, label_smoothing=0.1), axis=-1)
    return loss

# Define the model
inputs = Input(shape=(64, 128, 1))
x = inputs

# More Convolutional layers with increased filters
x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)

# LSTM with more units
outputs = []
for _ in range(3):  # Assuming 3 letters per image
    y = Reshape((512, 2))(x)  # Adjusted for larger dense output
    y = LSTM(256, return_sequences=True)(y)  # Increased LSTM units
    y = Flatten()(y)
    y = Dense(num_classes, activation='softmax')(y)
    outputs.append(y)

# Combine all outputs
outputs = Concatenate()(outputs)

model = Model(inputs=inputs, outputs=outputs)

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001, clipnorm=1.0), loss=custom_loss, metrics=['accuracy'])


# Learning rate scheduler and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience_lr, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=patience_es, restore_best_weights=True)

# Custom Callback for Visualization
class VisualizePredictionCallback(Callback):
    def __init__(self, validation_data, label_encoder, num_classes, sample_image_idx, visualize):
        self.validation_data = validation_data
        self.label_encoder = label_encoder
        self.num_classes = num_classes
        self.sample_image_idx = sample_image_idx
        self.visualize = visualize
        if self.visualize:
            self.figure, self.ax = plt.subplots(figsize=(6, 6))
            plt.ion()
            self.figure.show()
            self.figure.canvas.draw()

    def on_epoch_end(self, epoch, logs=None):
        if not self.visualize:
            return
        X_val, y_val = self.validation_data
        sample_image = X_val[self.sample_image_idx]
        true_label = y_val[self.sample_image_idx]
        
        sample_image = np.expand_dims(sample_image, axis=0)
        pred_probs = self.model.predict(sample_image)
        pred_probs = np.reshape(pred_probs, (-1, 3, self.num_classes))

        pred_labels = np.argmax(pred_probs, axis=-1).flatten()
        pred_labels_str = self.label_encoder.inverse_transform(pred_labels)
        
        true_label_indices = np.argmax(true_label.reshape(-1, 3, self.num_classes), axis=-1).flatten()
        true_label_str = self.label_encoder.inverse_transform(true_label_indices)

        self.ax.clear()
        self.ax.imshow(sample_image.squeeze(), cmap='gray')
        self.ax.set_title(f'Epoch {epoch + 1}\nPredicted: {pred_labels_str}\nTrue: {true_label_str}')
        self.ax.axis('off')

        self.figure.canvas.draw()
        plt.pause(1)

sample_image_idx = 2

visualize_callback = VisualizePredictionCallback(
    validation_data=(X_test, y_test),
    label_encoder=label_encoder,
    num_classes=num_classes,
    sample_image_idx=sample_image_idx,
    visualize=visualize
)

# Train the model with the new callback
callbacks = [reduce_lr, early_stopping]
if visualize:
    callbacks.append(visualize_callback)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_test, y_test),
    epochs=epochs,
    callbacks=callbacks
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show(block=True)
