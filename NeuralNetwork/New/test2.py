#importing libraries
import random
import numpy as np 

import matplotlib.pyplot as plt #for graphs
import os #for operating system dependent fucntionality
from keras import layers #for building layers of neural net
from keras.models import Model
from keras.models import load_model
from keras import callbacks #for training logs, saving to disk periodically
import cv2 #OpenCV(Open Source computer vision lib), containg CV algos
import string
data_dir = 'E:/Programming/PersonalProjects/CaptchaCoin/NeuralNetwork/base_images'


# Verify the file path
test_image_path = os.path.join(data_dir, 'EKF.png')  # Replace with an actual image name you know works
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

data_dir = 'E:/Programming/PersonalProjects/CaptchaCoin/NeuralNetwork/base_images'
print("Data directory:", data_dir)