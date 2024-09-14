
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'E:/Programming/PersonalProjects/CaptchaCoin/NeuralNetwork/captcha_images/train',
    target_size=(128, 64),
    batch_size=64,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    'E:/Programming/PersonalProjects/CaptchaCoin/NeuralNetwork/captcha_images/val',
    target_size=(128, 64),
    batch_size=64,
    class_mode='categorical'
)

# Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 64, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(36, activation='softmax')  # Assuming 36 possible CAPTCHA characters
])

# Compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Training
history = model.fit(
    train_gen,
    epochs=100,
    validation_data=val_gen,
    callbacks=[reduce_lr, early_stopping]
)

# Evaluate
loss, accuracy = model.evaluate(val_gen)
print(f'Test accuracy: {accuracy}')
