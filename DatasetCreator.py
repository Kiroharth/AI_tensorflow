import os
import random

def create_dataset_structure(base_path, classes):
    # Create the base directory if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Create train and validation directories
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'validation')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Create empty subdirectories for each class
    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

if __name__ == "__main__":
    # Extensive list of food categories
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

    create_dataset_structure(base_path, food_classes)
    print(f"Dataset structure created successfully with {len(food_classes)} food categories!")
