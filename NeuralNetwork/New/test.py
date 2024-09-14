import glob
from PIL import Image

data_dir = 'E:/Programming/PersonalProjects/CaptchaCoin/NeuralNetwork/base_images'

# Expected image size and label length
expected_size = (128, 64)
expected_label_length = 3

# Get list of all images and their corresponding labels
image_paths = sorted(list(glob.glob(f"{data_dir}/*.png")))
gt_list = [i.split('\\')[-1][:-4] for i in image_paths]  # Ground truth from file name

# Check for inconsistencies in image sizes and label lengths
for image_path, label in zip(image_paths, gt_list):
    # Check image size
    with Image.open(image_path) as img:
        size = img.size
        if size != expected_size:
            print(f"Image {image_path} has size {size}, expected {expected_size}")
    
    # Check label length
    if len(label) != expected_label_length:
        print(f"Inconsistent label length at {image_path}: {label}, length {len(label)}, expected {expected_label_length}")
