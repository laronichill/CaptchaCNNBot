import tensorflow as tf

# Check if TensorFlow is using the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Print the GPU device
print(tf.config.list_physical_devices('GPU'))