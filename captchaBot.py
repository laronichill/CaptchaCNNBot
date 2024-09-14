import pytesseract
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import base64
import os
from PIL import Image
from io import BytesIO
import string

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Set up Chrome options
options = Options()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Enable logging of console output
options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})

# Set up the WebDriver (make sure to specify the correct path to your WebDriver)
driver = webdriver.Chrome(options=options)

# Create a folder to save CAPTCHA images if it doesn't exist
output_folder = 'captcha_images'
os.makedirs(output_folder, exist_ok=True)

# To track the batch letter
batch_index = 0

driver.get('https://www.captchacoin.net/earn/mining/mine.html')
    
# Wait for the CAPTCHA elements to load
time.sleep(15)  # Adjust the sleep time as necessary

def process_captcha(batch_letter):
    
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
        return False  # Return false if there are JavaScript errors

    # Find the div with id "div"
    div_element = driver.find_element(By.ID, 'div')

    # Get all image elements inside this div
    img_elements = div_element.find_elements(By.TAG_NAME, 'img')
    print(f"Found {len(img_elements)} image elements inside the div.")

    for index, img in enumerate(img_elements):
        try:
            # Get the image data URL
            img_data_url = img.get_attribute('src')
            if img_data_url.startswith('data:image/png;base64,'):
                img_data = img_data_url.split('base64,')[1]
                image = Image.open(BytesIO(base64.b64decode(img_data)))
                # Save the image with the batch letter
                image_path = os.path.join(output_folder, f'captcha_{batch_letter}_{index}.png')
                image.save(image_path)
                print(f"Successfully saved image {index} as {image_path}.")
            else:
                print(f"Image source is not a base64 encoded PNG. Skipping image {index}.")
        except Exception as e:
            print(f"Error retrieving image from element {index}: {e}")

    return True

# Loop to retry CAPTCHA solving if incorrect
while True:
    batch_letter = string.ascii_lowercase[batch_index % len(string.ascii_lowercase)]
    success = process_captcha(batch_letter)
    if success:
        print(f"CAPTCHA images saved with batch letter {batch_letter}. Please manually solve the CAPTCHA.")
        batch_index += 1
        while True:
            user_input = input("Press 'r' and Enter to retry after solving the CAPTCHA: ")
            if user_input.strip().lower() == 'r':
                print("Retrying CAPTCHA solving...")
                break
    else:
        print("Retrying due to an error...")
        time.sleep(10)  # Wait before retrying, adjust as needed

# Close the driver
driver.quit()
