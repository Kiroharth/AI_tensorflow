import tensorflow_datasets as tfds
import os

# Get Food101 classes
food101_info = tfds.builder('food101').info
food101_classes = food101_info.features['label'].names

# Get Fruits-360 classes
fruits360_path = 'fruits-360_dataset_100x100/fruits-360/Training'
fruits360_classes = sorted(os.listdir(fruits360_path))

# Combine and sort all classes
all_classes = sorted(food101_classes + fruits360_classes)

# Print the total number of classes
print(f"Total number of classes: {len(all_classes)}")

# Print all classes
for i, class_name in enumerate(all_classes, 1):
    print(f"{i}. '{class_name}'")

# Optionally, save to a file
with open('food_classes.txt', 'w') as f:
    for class_name in all_classes:
        f.write(f"'{class_name}',\n")
