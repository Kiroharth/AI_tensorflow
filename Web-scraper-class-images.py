import os
import requests
from bs4 import BeautifulSoup
import time
import random
import numpy as np
from PIL import Image
from io import BytesIO
import logging
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Pixabay API key
PIXABAY_API_KEY = 'your_pixabay_api_key'

# Set up logging
logging.basicConfig(filename='image_downloader.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Batch size for image verification
BATCH_SIZE = 8

def create_dataset_structure(base_path, classes):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'validation')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

# Batch verification function
def batch_verify_images(img_data_list, target_class):
    """
    This function verifies a batch of images using the ResNet50 model.
    """
    try:
        # Preprocess all images in the batch
        imgs_preprocessed = []
        for img_data in img_data_list:
            img = Image.open(BytesIO(img_data)).convert('RGB')
            img = img.resize((224, 224))
            x = img_to_array(img)
            x = preprocess_input(x)
            imgs_preprocessed.append(x)
        
        # Stack them into a single batch
        batch = np.stack(imgs_preprocessed)
        
        # Make predictions for the entire batch
        preds = model.predict(batch)
        decoded_preds = decode_predictions(preds, top=5)
        
        # Verify if any image in the batch matches the target class
        verified_images = []
        for i, decoded in enumerate(decoded_preds):
            if any(pred_class.lower() in target_class.lower() for _, pred_class, score in decoded):
                verified_images.append(img_data_list[i])
        
        return verified_images
    except Exception as e:
        logging.error(f"Error in batch image verification: {str(e)}")
        return []

def count_existing_images(folder):
    return len([f for f in os.listdir(folder) if f.endswith('.jpg')])

def download_images_from_url(img_url, query, output_folder, img_name):
    try:
        img_data = requests.get(img_url).content
        with open(os.path.join(output_folder, img_name), 'wb') as handler:
            handler.write(img_data)
        logging.info(f"Downloaded: {img_name} to {output_folder}")
        return img_data
    except Exception as e:
        logging.error(f"Error downloading image from {img_url}: {str(e)}")
        return None

# Pixabay API downloader with retries
def download_from_pixabay(query, num_images, retries=3):
    try:
        url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={query}&image_type=photo&per_page={num_images}"
        response = requests.get(url)
        if response.status_code == 429:  # Too many requests, handle rate limiting
            raise Exception("Rate limit reached for Pixabay API")
        response.raise_for_status()
        return [img['largeImageURL'] for img in response.json().get('hits', [])]
    except Exception as e:
        if retries > 0:
            logging.warning(f"Retrying Pixabay API for {query}, {retries} attempts left: {str(e)}")
            time.sleep(2 ** (3 - retries))  # Exponential backoff
            return download_from_pixabay(query, num_images, retries - 1)
        logging.error(f"Error fetching Pixabay images for {query}: {str(e)}")
        return []

# Function to fetch images from Google Images with basic error handling
def fetch_google_images(query, num_images):
    try:
        url = f"https://www.google.com/search?q={query}&tbm=isch"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tags = soup.find_all('img')
        return [img.get('src') for img in img_tags[:num_images] if img.get('src')]
    except Exception as e:
        logging.error(f"Error fetching Google images for {query}: {str(e)}")
        return []

# Updated download_images function: Prioritize Pixabay, fallback to Google Images
def download_images(query, num_images, train_folder, val_folder):
    existing_train = count_existing_images(train_folder)
    existing_val = count_existing_images(val_folder)

    if existing_train >= num_images and existing_val >= num_images:
        logging.info(f"Skipping {query}: already have sufficient images")
        return

    num_to_download_train = num_images - existing_train
    num_to_download_val = num_images - existing_val
    total_images_needed = num_to_download_train + num_to_download_val

    # 1. First try downloading from Pixabay
    pixabay_images = download_from_pixabay(query, total_images_needed)
    
    # 2. If not enough images, fallback to Google Images
    if len(pixabay_images) < total_images_needed:
        google_images = fetch_google_images(query, total_images_needed - len(pixabay_images))
        pixabay_images.extend(google_images)

    # Multithreading for downloading images concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        count_train = 0
        count_val = 0
        img_data_list = []  # To store image data for batch verification

        for img_url in pixabay_images:
            if count_train >= num_to_download_train and count_val >= num_to_download_val:
                break

            img_name = f"{query}_{existing_train + count_train + existing_val + count_val}.jpg"
            if count_train < num_to_download_train:
                output_folder = train_folder
                count_train += 1
            else:
                output_folder = val_folder
                count_val += 1

            futures.append(executor.submit(download_images_from_url, img_url, query, output_folder, img_name))
        
        for future in as_completed(futures):
            img_data = future.result()
            if img_data:
                img_data_list.append(img_data)

            # Batch process the image verification once enough images are collected
            if len(img_data_list) >= BATCH_SIZE or (count_train + count_val) >= total_images_needed:
                verified_images = batch_verify_images(img_data_list, query)
                
                # Save the verified images
                for verified_img_data in verified_images:
                    img_name = f"{query}_{existing_train + count_train + existing_val + count_val}.jpg"
                    output_folder = train_folder if random.random() < 0.8 else val_folder
                    with open(os.path.join(output_folder, img_name), 'wb') as handler:
                        handler.write(verified_img_data)
                    logging.info(f"Verified and saved: {img_name} to {output_folder}")
                    img_data_list.remove(verified_img_data)  # Remove the processed image from the batch

    logging.info(f"Downloaded {count_train} images for training and {count_val} images for validation for {query}")

# Example usage
food_classes = [
    # Fruits, Vegetables, Meats, etc.
    "apple", "banana", "orange", "strawberry", "grape", "pineapple", "mango", "watermelon", "kiwi", "peach",
    # ... (rest of the classes)
]
base_path = "#Custom-Food-Dataset"
num_images_per_class = 500

create_dataset_structure(base_path, food_classes)

for food_class in food_classes:
    train_folder = os.path.join(base_path, 'train', food_class)
    val_folder = os.path.join(base_path, 'validation', food_class)
    download_images(f"{food_class} food", num_images_per_class, train_folder, val_folder)
    time.sleep(random.uniform(1, 3))
