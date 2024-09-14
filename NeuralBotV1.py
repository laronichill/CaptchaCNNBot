import os
import base64
import time
import numpy as np
import pytesseract
from PIL import Image
from io import BytesIO
import requests  # Ensure you have requests installed
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import string
from sklearn.preprocessing import LabelEncoder  # Add this import

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Path to the ChromeDriver executable
chrome_driver_path = 'C:/path/to/chromedriver.exe'  # Update this path

# Set up Chrome options
options = Options()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})

# Set up the WebDriver
driver = webdriver.Chrome(options=options)

# Load the trained model
model = load_model('captcha_solver_model.keras')

# Initialize and fit the LabelEncoder with uppercase letters
labels = list(string.ascii_uppercase)  # Use uppercase letters
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Function to preprocess images for model prediction
def preprocess_image(image):
    image = image.convert('L')  # Convert image to grayscale
    image = image.resize((128, 64))  # Resize image to the input size expected by the model
    image_array = img_to_array(image) / 255.0
    return image_array

# Function to predict CAPTCHA text using the model
def predict_captcha(image):
    image_array = preprocess_image(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    prediction = model.predict(image_array)
    
    # Assuming the model predicts the entire CAPTCHA string at once
    predicted_labels = [label_encoder.inverse_transform([np.argmax(p)]) for p in prediction[0]]
    return ''.join([label[0] for label in predicted_labels])

# Function to process and predict CAPTCHA images
def process_captcha():
    try:
        # Retrieve console logs
        logs = driver.get_log('browser')
        for log in logs:
            print(log)

        # Check for JavaScript errors
        js_errors = [log['message'] for log in logs if 'ERROR' in log['level']]
        if js_errors:
            print("JavaScript errors detected:")
            for error in js_errors:
                print(error)
            return False

        # Find the div with id "div"
        div_element = driver.find_element(By.ID, 'div')

        # Get all image elements inside this div that are used for CAPTCHA
        img_elements = div_element.find_elements(By.XPATH, ".//img")  # Only select <img> elements
        print(f"Found {len(img_elements)} image elements inside the div.")

        for index, img in enumerate(img_elements):
            try:
                # Get the image source
                img_src = img.get_attribute('src')
                if img_src.startswith('data:image/png;base64,'):
                    img_data = img_src.split('base64,')[1]
                    image = Image.open(BytesIO(base64.b64decode(img_data)))
                elif img_src.startswith('http') or img_src.startswith('https'):
                    # Handle images directly accessible via URL
                    image = Image.open(BytesIO(requests.get(img_src).content))
                else:
                    print(f"Image source is not recognized. Skipping image {index}.")
                    continue

                # Predict the text of the CAPTCHA image
                captcha_text = predict_captcha(image)
                print(f"Predicted CAPTCHA text for image {index}: {captcha_text}")
            except Exception as e:
                print(f"Error processing image from element {index}: {e}")

        return True
    except Exception as e:
        print(f"Error in process_captcha: {e}")
        return False

# Main loop to navigate and process CAPTCHA images
try:
    driver.get('https://www.captchacoin.net/earn/mining/mine.html')
    
    # Wait until a specific element is present to confirm the page has loaded
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, 'div')))
    print("Page loaded successfully.")
    
    while True:
        success = process_captcha()
        if success:
            while True:
                user_input = input("Press 'r' and Enter to retry after solving the CAPTCHA: ")
                if user_input.strip().lower() == 'r':
                    print("Retrying CAPTCHA solving...")
                    break
        else:
            print("Retrying due to an error...")
            time.sleep(10)
except Exception as e:
    print(f"Error during navigation or processing: {e}")
finally:
    # Ensure the driver is closed properly
    driver.quit()
