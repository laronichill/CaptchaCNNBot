#importing libraries
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import string
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Set variables
data_dir = 'E:/Programming/PersonalProjects/CaptchaCoin/dataImages'
varEpoch = 200
varBatchSize = 128
varLearningRate = 0.001
varNewTrainingOnly = True
var_Dropout1 = 0.9
var_Dropout2 = 0.7

varNumOfData = 3072

# Callbacks for learning rate reduction and early stopping
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8,
    min_lr=1e-6
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=16,
    restore_best_weights=True
)

imgshape = (64, 128, 1)  # 64-height, 128-width, 1-channel for grayscale

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

# Print available images in dataset directory
# print(os.listdir(data_dir))

# Total number of images in dataset
n = len(os.listdir(data_dir))

# Compute class weights
def compute_class_weights(y, n_classes):
    y_integers = np.argmax(y.reshape(-1, n_classes), axis=1)
    unique_labels = np.unique(y_integers)
    
    # Compute weights for the present labels only
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=y_integers)
    
    # Create a dictionary to map all class indices
    class_weights_dict = {i: 0 for i in range(n_classes)}
    for i, label in enumerate(unique_labels):
        class_weights_dict[label] = class_weights[i]
    
    # Convert dictionary to tensor
    class_weights_tensor = tf.constant([class_weights_dict[i] for i in range(n_classes)], dtype=tf.float32)
    
    return class_weights_tensor

def extract_characters(data_dir):
    characters_set = set()  # Use a set to avoid duplicates

    for pic in os.listdir(data_dir):
        # Check if the file is a PNG image
        if pic.endswith('.png'):
            # Extract the first 3 characters of the filename
            pic_target = pic[:3]
            
            # Add each character to the set
            for char in pic_target:
                if char in string.ascii_uppercase:  # Ensure it's an uppercase letter
                    characters_set.add(char)
    
    # Convert the set to a sorted list
    characters_string = ''.join(sorted(characters_set))
    return characters_string

# Characters that can be found in the CAPTCHA
# character = string.ascii_uppercase + '0'

character = extract_characters(data_dir) + '0'
nchar = len(character)

print(f"{nchar} : {character}")

# Preprocess the image data
def preprocess():
    X = np.zeros((n, 64, 128, 1))
    y = np.zeros((3, n, nchar))  # Now nchar = 27 to include dummy class

    for i, pic in enumerate(os.listdir(data_dir)):
        img = cv2.imread(os.path.join(data_dir, pic), cv2.IMREAD_GRAYSCALE)
        pic_target = pic[:3]  # Extract the first 3 letters (CAPTCHA part)
        img_num = int(pic[-5])  # Extract the number part (0 to 7)

        img = img / 255.0  # Normalize pixel values
        img = np.reshape(img, (64, 128, 1))  # Reshape to (64, 128, 1) for grayscale

        if img_num == 0:
            # For _0 images, assign correct labels
            target = np.zeros((3, nchar))
            for j, k in enumerate(pic_target):
                index = character.find(k)
                target[j, index] = 1
            y[:, i] = target  # Assign the correct target labels
        else:
            # For _1 to _7 images, assign "dummy" labels ('D')
            dummy_index = character.find('0')
            y[:, i] = np.zeros((3, nchar))
            y[:, i, dummy_index] = 1  # All positions are assigned to "dummy" class

        X[i] = img

    return X, y

# Preprocess the data
X, y = preprocess()

# Split data into training and testing sets
X_train, y_train = X[:varNumOfData], y[:, :varNumOfData]
X_test, y_test = X[varNumOfData:], y[:, varNumOfData:]

print("Unique labels in y_train[0]:", np.unique(np.argmax(y_train[0], axis=1)))
expected_labels = set(range(nchar))
present_labels = set(np.unique(np.argmax(y_train[0], axis=1)))
missing_labels = expected_labels - present_labels
print(f"Missing labels: {missing_labels}")

# Compute class weights for each output
class_weights_letter1 = compute_class_weights(y_train[0], nchar)
class_weights_letter2 = compute_class_weights(y_train[1], nchar)
class_weights_letter3 = compute_class_weights(y_train[2], nchar)

# Prepare class weights as a tensor
class_weights_dict = {
    'letter0': class_weights_letter1,
    'letter1': class_weights_letter2,
    'letter2': class_weights_letter3
}

# Create the model
def createmodel():
    img = layers.Input(shape=imgshape)
    conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img)
    mp1 = layers.MaxPooling2D(padding='same')(conv1)
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    mp2 = layers.MaxPooling2D(padding='same')(conv2)
    conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)
    bn = layers.BatchNormalization()(conv3)
    mp3 = layers.MaxPooling2D(padding='same')(bn)
    
    flat = layers.Flatten()(mp3)
    
    outs = []
    for i in range(3):  # 3 characters to predict
        dens1 = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01))(flat)
        res = layers.Dense(nchar, activation='softmax', name=f'letter{i}')(dens1)
        outs.append(res)
    
    model = Model(img, outs)

    def weighted_loss(y_true, y_pred, class_weights):
        weights = tf.gather(class_weights, tf.argmax(y_true, axis=-1))
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        weighted_loss = loss * weights
        return tf.reduce_mean(weighted_loss)

    model.compile(
        loss=lambda y_true, y_pred: weighted_loss(y_true, y_pred, class_weights_dict['letter0']),
        optimizer=Adam(learning_rate=varLearningRate),
        metrics={'letter0': 'accuracy', 'letter1': 'accuracy', 'letter2': 'accuracy'},
    )
    return model

# Load or create a new model
if varNewTrainingOnly:
    model = createmodel()
else:
    model = load_model('E:/Programming/PersonalProjects/CaptchaCoin/NeuralNetwork/CNN Online/best_model.keras')

# Create a generator for multi-label data
def multi_label_generator(X, y, batch_size, datagen):
    image_generator = datagen.flow(X, batch_size=batch_size, shuffle=True)
    
    while True:
        X_batch = next(image_generator)
        if X_batch.shape[-1] == 3:
            X_batch = np.mean(X_batch, axis=-1, keepdims=True)
        
        idx = np.random.choice(len(y[0]), size=min(batch_size, len(y[0])), replace=False)
        y1_batch = y[0][idx]
        y2_batch = y[1][idx]
        y3_batch = y[2][idx]
        
        X_batch = X_batch.astype(np.float32)
        y1_batch = y1_batch.astype(np.float32)
        y2_batch = y2_batch.astype(np.float32)
        y3_batch = y3_batch.astype(np.float32)
        
        yield X_batch, (y1_batch, y2_batch, y3_batch)

callbacks = [reduce_lr, early_stopping]

dataset = tf.data.Dataset.from_generator(
    lambda: multi_label_generator(X_train, y_train, varBatchSize, datagen),
    output_signature=( 
        tf.TensorSpec(shape=(None, 64, 128, 1), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(None, nchar), dtype=tf.float32),
            tf.TensorSpec(shape=(None, nchar), dtype=tf.float32),
            tf.TensorSpec(shape=(None, nchar), dtype=tf.float32),
        )
    )
).prefetch(tf.data.AUTOTUNE)

validation_data = tf.data.Dataset.from_generator(
    lambda: multi_label_generator(X_test, y_test, varBatchSize, datagen),
    output_signature=( 
        tf.TensorSpec(shape=(None, 64, 128, 1), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(None, nchar), dtype=tf.float32),
            tf.TensorSpec(shape=(None, nchar), dtype=tf.float32),
            tf.TensorSpec(shape=(None, nchar), dtype=tf.float32),
        )
    )
).prefetch(tf.data.AUTOTUNE)

# Train the model
hist = model.fit(
    dataset,
    steps_per_epoch=len(X_train) // varBatchSize,
    epochs=varEpoch,
    validation_data=validation_data,
    callbacks=callbacks
)
