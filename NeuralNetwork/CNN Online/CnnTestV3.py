#importing libraries
import random
import keras
import numpy as np 

import matplotlib.pyplot as plt #for graphs
import tensorflow as tensorflow
import os #for operating system dependent fucntionality
from tensorflow.keras import layers #for building layers of neural net
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import callbacks #for training logs, saving to disk periodically
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

import cv2 #OpenCV(Open Source computer vision lib), containg CV algos
import string

data_dir = 'E:/Programming/PersonalProjects/CaptchaCoin/NeuralNetwork/base_images'
varEpoch = 50
varBatchSize = 128
varLearningRate = 0.001
varNewTrainingOnly = True
var_Dropout1 = 0.8
var_Dropout2 = 0.5
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

#print images in dataset
os.listdir(data_dir)

#total no of images in dataset
n=len(os.listdir(data_dir))

#defining size of image
imgshape=(64,128,1) #64-height, 128-width, 1-no of channels

character= string.ascii_uppercase # All symbols captcha can contain
nchar = len(character) #total number of char possible

print(character)

#preprocesss image
def preprocess():
  X = np.zeros((n,64,128,1)) #1070*64*128 array with all entries 0
  y = np.zeros((3,n,nchar)) #3*1070*36(3 letters in captcha) with all entries 0

  for i, pic in enumerate(os.listdir(data_dir)):
  #i represents index no. of image in directory 
  #pic contains the file name of the particular image to be preprocessed at a time
    
    img = cv2.imread(os.path.join(data_dir, pic), cv2.IMREAD_GRAYSCALE) #Read image in grayscale format
    pic_target = pic[:-4]#this drops the .png extension from file name and contains only the captcha for training
    
    if len(pic_target) < 4: #captcha is not more than 3 letters
      img = img / 255.0 #scales the image between 0 and 1
      img = np.reshape(img, (64, 128, 1)) #reshapes image to width 128 , height 64 ,channel 1 

      target=np.zeros((3,nchar)) #creates an array of size 5*36 with all entries 0

      for j, k in enumerate(pic_target):
      #j iterates from 0 to 4(5 letters in captcha)
      #k denotes the letter in captcha which is to be scanned
         index = character.find(k) #index stores the position of letter k of captcha in the character string
         target[j, index] = 1 #replaces 0 with 1 in the target array at the position of the letter in captcha

      X[i] = img #stores all the images
      y[:,i] = target #stores all the info about the letters in captcha of all images

  return X,y

#create model
def createmodel():
    img = layers.Input(shape=imgshape)
    conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img)
    mp1 = layers.MaxPooling2D(padding='same')(conv1)
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    mp2 = layers.MaxPooling2D(padding='same')(conv2)
    conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)
    # drop_conv3 = layers.Dropout(var_Dropout1)(conv3) 
    bn = layers.BatchNormalization()(conv3)
    mp3 = layers.MaxPooling2D(padding='same')(bn)
    
    flat = layers.Flatten()(mp3)
    
    outs = []
    for i in range(3):  # Assuming 3 output layers
        dens1 = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01))(flat)  # Increase L2 penalty

        # dens1 = layers.Dense(64, activation='relu')(flat)
        drop = layers.Dropout(var_Dropout2)(dens1)
        res = layers.Dense(nchar, activation='softmax', name=f'letter{i}')(drop)  # Named outputs
        outs.append(res)
    
    model = Model(img, outs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=varLearningRate),
                  metrics={f'letter{i}': 'accuracy' for i in range(3)})  # Use correct names
                  # metrics={f'letter{i}': ['accuracy', 'Precision'] for i in range(3)})
    return model
  

  
  
#Which model to use model
while True:
    if varNewTrainingOnly:
        model = createmodel()
        break
    
    user_input = input('New model training session? Yes / No: ')
    
    if user_input.capitalize() == 'Yes':
        model = createmodel()
        break
    elif user_input.capitalize() == 'No':
        model = load_model('E:/Programming/PersonalProjects/CaptchaCoin/NeuralNetwork/CNN Online/best_model.keras')
        break
    else:
        print('Enter Yes or No')
        continue

X,y=preprocess()
#split the 1070 samples where 970 samples will be used for training purpose
X_train, y_train = X[:2048], y[:, :2048]
X_test, y_test = X[2048:], y[:, 2048:]

""" datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)
train_generator = datagen.flow(
    X_train, [y_train[0], y_train[1], y_train[2]],
    batch_size=varBatchSize
)

callbacks = [reduce_lr, early_stopping]

#Applying the model
hist = model.fit(train_generator, batch_size=varBatchSize, epochs=varEpoch, validation_split=0.2, callbacks=callbacks) """

callbacks = [reduce_lr, early_stopping]
#Applying the model
hist = model.fit(X_train, [y_train[0], y_train[1], y_train[2]], batch_size=varBatchSize, epochs=varEpoch, validation_split=0.2, callbacks=callbacks)

#batch size- 32 defines no. of samples per gradient update
#Validation split=0.2 splits the training set in 80-20% for training nd testing

#graph of loss vs epochs
""" for label in ["loss"]:
  plt.plot(hist.history[label],label=label)
  plt.legend()
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.show() """

#Loss on training set
#Finding Loss on training set
preds = model.evaluate(X_train, [y_train[0], y_train[1], y_train[2]])
print ("Loss on training set= " + str(preds[0]))

#Finding loss on test set
preds = model.evaluate(X_test, [y_test[0], y_test[1], y_test[2]])
print ("Loss on testing set= " + str(preds[0]))

# Plotting the loss
def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Plotting the accuracy for each letter
def plot_accuracy(history):
    for i in range(3):  # For each letter (assuming 3 letters)
        plt.plot(history.history[f'letter{i}_accuracy'], label=f'Training Accuracy for letter {i+1}')
        plt.plot(history.history[f'val_letter{i}_accuracy'], label=f'Validation Accuracy for letter {i+1}', linestyle='--')
    plt.title('Training and Validation Accuracy per Letter')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Add this after your training completes to display the graphs
plot_loss(hist)
plot_accuracy(hist)


#to predict captcha
def predict(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        img = img / 255.0
        img = np.reshape(img, (64, 128, 1))  # Ensure correct input shape
        img = np.expand_dims(img, axis=0)    # Add batch dimension

        try:
            res = model.predict(img)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

        result = [np.reshape(r, (26,)) for r in res]  # Ensure each output is reshaped correctly
        k_ind = [np.argmax(r) for r in result]
        capt = ''.join(character[k] for k in k_ind)
        return capt
    else:
        print("Image not detected")
        return None


data_dir = 'E:/Programming/PersonalProjects/CaptchaCoin/NeuralNetwork/base_images'
random_file = os.path.join(data_dir, 'EKF.png')  # Replace with an actual image name you know works

print("Captcha:" +  random_file + " | Predicted:", predict(random_file))

model.save('E:/Programming/PersonalProjects/CaptchaCoin/NeuralNetwork/CNN Online/best_model.keras')