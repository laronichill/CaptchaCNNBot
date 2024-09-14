import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Bidirectional, LSTM, Reshape, TimeDistributed, Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2

# Variables
data_dir = 'E:/Programming/PersonalProjects/CaptchaCoin/NeuralNetwork/base_images'
img_width = 128
img_height = 64
batch_size = 128  # Do not change
learning_rate = 0.005
epochs = 100
patience_lr = 8
patience_es = 24
dropout_var = 0.2
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
y_one_hot = y_one_hot.reshape(-1, 3, num_classes)

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

# Define the model
inputs = Input(shape=(img_height, img_width, 1), name="input_layer")

# Convolutional layers
x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
x = MaxPooling2D((2, 2))(x)

# Flatten and reshape for LSTM input
x = Reshape((-1, 512))(x)  # Reshape to (batch_size, time_steps, features)

# LSTM layers
x = Bidirectional(LSTM(128, return_sequences=True, dropout=dropout_var))(x)
x = Bidirectional(LSTM(64, return_sequences=True, dropout=dropout_var))(x)

# Dense layer without softmax (CTC loss will handle this)
outputs = Dense(num_classes + 1, activation=None)(x)  # +1 for CTC blank label

# Define the CTC loss function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    
    # Ensure input_length and label_length are squeezed properly
    input_length = K.cast(K.squeeze(input_length, axis=-1), dtype='int32')  
    label_length = K.cast(K.squeeze(label_length, axis=-1), dtype='int32')  
    
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# Create labels, input_length, and label_length as inputs
labels = Input(name='the_labels', shape=[3], dtype='float32')  # 3 characters in each label
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=(1,), dtype='int64')  # Correct shape

# Use Lambda layer to wrap CTC loss function
ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

# Define the model
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=ctc_loss)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss={'ctc': lambda y_true, y_pred: y_pred})

# Data generator for CTC
def data_generator(X, y, batch_size):
    while True:
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            input_length_batch = np.ones((len(X_batch), 1)) * (img_width // 4)
            label_length_batch = np.ones((len(X_batch), 1)) * 3  # Because each CAPTCHA has exactly 3 characters
            inputs = {
                'input_layer': X_batch,
                'the_labels': np.argmax(y_batch, axis=-1),  # Convert one-hot to class indices
                'input_length': input_length_batch,
                'label_length': label_length_batch
            }
            
            outputs = {'ctc': np.zeros([len(X_batch)])}  # Dummy variable for the CTC loss
            yield inputs, outputs


# Learning rate scheduler and early stopping
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.75,  # Reduces learning rate by 25%
    patience=patience_lr, 
    min_lr=1e-6
)
early_stopping = EarlyStopping(monitor='val_loss', patience=patience_es, restore_best_weights=True)

# Train the model
callbacks = [reduce_lr, early_stopping]

history = model.fit(
    data_generator(X_train, y_train, batch_size),
    validation_data=data_generator(X_test, y_test, batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    validation_steps=len(X_test) // batch_size,
    epochs=epochs,
    callbacks=callbacks
)

# Evaluate the model by decoding predictions
from tensorflow.keras.backend import ctc_decode

# Decode the predictions
def decode_batch(pred):
    decoded, _ = ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)
    return K.get_value(decoded[0])

# Predict on a test sample
pred = model.predict([X_test, np.ones(len(X_test)) * (img_width // 4), np.ones(len(X_test)) * 3])
decoded_pred = decode_batch(pred)

# Convert to string labels
for seq in decoded_pred:
    print("Predicted label:", ''.join([label_encoder.inverse_transform([int(i)])[0] for i in seq if i != -1]))

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show(block=True)
