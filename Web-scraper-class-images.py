import os
import requests
from bs4 import BeautifulSoup
import time
import random
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

def create_dataset_structure(base_path, classes):
    # Create the base directory if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Create train and validation directories
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'validation')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Create subdirectories for each class
    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

def verify_image(img_data, target_class):
    try:
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img = img.resize((224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        decoded_preds = decode_predictions(preds, top=5)[0]

        for _, pred_class, score in decoded_preds:
            if pred_class.lower() in target_class.lower():
                return True
        return False
    except Exception as e:
        print(f"Error verifying image: {str(e)}")
        return False

def count_existing_images(folder):
    return len([f for f in os.listdir(folder) if f.endswith('.jpg')])

def download_images(query, num_images, train_folder, val_folder):
    existing_train = count_existing_images(train_folder)
    existing_val = count_existing_images(val_folder)
    total_existing = existing_train + existing_val
    
    if total_existing >= num_images:
        print(f"Skipping {query}: already have {total_existing} images")
        return

    num_to_download = num_images - total_existing
    
    url = f"https://www.google.com/search?q={query}&tbm=isch"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')

    count = 0
    for img in img_tags:
        if count >= num_to_download:
            break

        img_url = img.get('src')
        if img_url and img_url.startswith('http'):
            try:
                img_data = requests.get(img_url).content
                
                if not verify_image(img_data, query):
                    print(f"Skipping image: not verified as {query}")
                    continue

                img_name = f"{query}_{total_existing + count}.jpg"
                
                if random.random() < 0.8:
                    output_folder = train_folder
                else:
                    output_folder = val_folder
                
                with open(os.path.join(output_folder, img_name), 'wb') as handler:
                    handler.write(img_data)
                print(f"Downloaded and verified: {img_name} to {output_folder}")
                count += 1
                
                time.sleep(random.uniform(0.5, 1.5))
            except Exception as e:
                print(f"Error downloading {img_url}: {str(e)}")

    print(f"Downloaded {count} additional verified images for {query}")
# Example usage
food_classes = [
        # Fruits
        "apple", "banana", "orange", "strawberry", "grape", "pineapple", "mango", "watermelon", "kiwi", "peach",
        "pear", "plum", "cherry", "blueberry", "raspberry", "blackberry", "lemon", "lime", "coconut", "fig",
        "pomegranate", "papaya", "guava", "passion_fruit", "dragonfruit", "lychee", "apricot", "cantaloupe", "honeydew",
        # Vegetables
        "carrot", "broccoli", "tomato", "cucumber", "lettuce", "potato", "onion", "bell_pepper", "spinach", "corn",
        "pea", "green_bean", "cauliflower", "asparagus", "eggplant", "zucchini", "pumpkin", "squash", "radish", "beet",
        "celery", "mushroom", "garlic", "ginger", "cabbage", "kale", "brussels_sprout", "artichoke", "leek",
        # Meats
        "chicken", "beef", "pork", "lamb", "turkey", "salmon", "tuna", "shrimp", "bacon", "sausage",
        "ham", "steak", "meatball", "duck", "goose", "venison", "rabbit", "quail", "veal", "liver",
        # Seafood
        "cod", "haddock", "trout", "sardine", "anchovy", "crab", "lobster", "mussel", "clam", "oyster",
        # Dairy
        "cheese", "milk", "yogurt", "butter", "cream", "ice_cream", "sour_cream", "cottage_cheese", "whipped_cream",
        # Grains
        "bread", "rice", "pasta", "cereal", "oatmeal", "quinoa", "couscous", "barley", "rye", "millet",
        "bulgur", "farro", "cornmeal", "buckwheat", "wheat", "noodle",
        # Desserts
        "cake", "cookie", "pie", "chocolate", "candy", "donut", "ice_cream", "pudding", "brownie", "cupcake",
        "muffin", "croissant", "eclair", "macaroon", "tiramisu", "cheesecake", "tart", "gelato", "sorbet",
        # Beverages
        "coffee", "tea", "soda", "juice", "water", "smoothie", "milkshake", "lemonade", "wine", "beer",
        "cocktail", "whiskey", "vodka", "gin", "rum", "tequila", "champagne",
        # Fast Food
        "pizza", "hamburger", "french_fries", "hot_dog", "taco", "burrito", "sandwich", "fried_chicken", "nugget",
        # International Cuisines
        "sushi", "curry", "pasta", "taco", "kebab", "paella", "ramen", "pho", "pad_thai", "sashimi",
        "dim_sum", "falafel", "hummus", "gyro", "pierogi", "goulash", "risotto", "moussaka", "bibimbap",
        # Condiments and Sauces
        "ketchup", "mustard", "mayonnaise", "salsa", "soy_sauce", "hot_sauce", "bbq_sauce", "ranch_dressing",
        "vinaigrette", "tartar_sauce", "pesto", "guacamole", "aioli", "chutney", "wasabi",
        # Nuts and Seeds
        "almond", "peanut", "walnut", "sunflower_seed", "pumpkin_seed", "cashew", "pistachio", "pecan", "hazelnut",
        # Herbs and Spices
        "basil", "oregano", "cinnamon", "pepper", "garlic", "ginger", "thyme", "rosemary", "sage", "mint",
        "paprika", "cumin", "turmeric", "nutmeg", "cardamom", "clove", "saffron", "vanilla",
        # Breakfast Items
        "pancake", "waffle", "french_toast", "bagel", "muesli", "granola", "bacon", "sausage", "hash_brown",
        # Snacks
        "chips", "popcorn", "pretzel", "cracker", "nachos", "trail_mix", "jerky", "granola_bar", "energy_bar"
    ]
base_path = "#Custom-Food-Dataset"
num_images_per_class = 500

create_dataset_structure(base_path, food_classes)

for food_class in food_classes:
    train_folder = os.path.join(base_path, 'train', food_class)
    val_folder = os.path.join(base_path, 'validation', food_class)
    download_images(f"{food_class} food", num_images_per_class, train_folder, val_folder)
    time.sleep(random.uniform(1, 3))
