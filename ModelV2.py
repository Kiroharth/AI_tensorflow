# %%
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
import fiftyone as fo
import fiftyone.zoo as foz
import os
import logging
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import requests
import psutil
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.keras import mixed_precision

# %% [markdown]
# #### Using GPU

# %%
# Limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# %% [markdown]
# **Logging setup**

# %%
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
os.environ["FIFTYONE_DEFAULT_DATASET_DIR"] = "/datasets"
fo.config.dataset_zoo_dir = "/datasets"

# %% [markdown]
# ### Data augmentation and preprocessing

# %%
def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image


# %%
def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image = augment_image(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

# %%
def prepare_dataset(dataset, batch_size=8, shuffle_buffer=1000):
    dataset = dataset.cache()
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# %% [markdown]
# ### Load Model

# %% [markdown]
# **Load Food101**

# %%
food101_train = tfds.load('food101', split='train', as_supervised=True)
food101_val = tfds.load('food101', split='validation', as_supervised=True)

# %% [markdown]
# **Load Open Images V7**

# %%

def load_and_preprocess_image(path, label):
    try:
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (224, 224))
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return img, label
    except tf.errors.InvalidArgumentError as e:
        logging.error(f"Error processing image at path {path}: {str(e)}")
        return None, None

def fiftyone_to_tf_dataset(fo_dataset):
    image_paths = []
    labels = []
    total_samples = 0
    skipped_samples = 0

    for sample in fo_dataset.iter_samples():
        total_samples += 1
        
        # Check for labels in different possible fields
        label = None
        if hasattr(sample, 'ground_truth'):
            label = sample.ground_truth.label
        elif hasattr(sample, 'positive_labels') and sample.positive_labels:
            if sample.positive_labels.classifications:
                label = sample.positive_labels.classifications[0].label
        elif hasattr(sample, 'detections') and sample.detections:
            if sample.detections.detections:
                label = sample.detections.detections[0].label
        
        if label in food_classes:
            image_paths.append(sample.filepath)
            labels.append(food_classes.index(label))
        else:
            skipped_samples += 1
            continue

    print(f"Total samples: {total_samples}")
    print(f"Skipped samples: {skipped_samples}")
    print(f"Processed samples: {len(image_paths)}")

    if not image_paths:
        raise ValueError("No samples matched the criteria. Check your food_classes and dataset labels.")

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda x, y: x is not None and y is not None)
    return dataset


# %%
food_classes = ["Food", "Egg (Food)", "Fast food", "Seafood"]

# %%
open_v7_train = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    classes=food_classes,
    max_samples=100000,  # Adjust as needed if we want more Samples
    only_matching=True,
    drop_existing_dataset=True,
)

# Print some information about the dataset
print(f"Dataset info: {open_v7_train}")
print(f"Number of samples: {len(open_v7_train)}")
print(f"Sample fields: {open_v7_train.first().field_names}")


print("Sample of labels in the dataset: ")
for sample in open_v7_train.take(10):  # Limit to 10 samples for brevity
    if hasattr(sample, 'ground_truth'):
        print(sample.ground_truth.label)
    elif hasattr(sample, 'positive_labels') and sample.positive_labels:
        if sample.positive_labels.classifications:
            print(sample.positive_labels.classifications[0].label)
    elif hasattr(sample, 'detections') and sample.detections:
        if sample.detections.detections:
            print(sample.detections.detections[0].label)
    else:
        print("No label found")

# Convert to TensorFlow dataset
open_v7_train_tf = fiftyone_to_tf_dataset(open_v7_train)

# %%
open_v7_val = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    classes=food_classes,
    max_samples=10000,  # Adjust
    only_matching=True,
    drop_existing_dataset=True,
)

# Print some information about the dataset
print(f"Dataset info: {open_v7_val}")
print(f"Number of samples: {len(open_v7_val)}")
print(f"Sample fields: {open_v7_val.first().field_names}")

# Convert to TensorFlow dataset
open_v7_val_tf = fiftyone_to_tf_dataset(open_v7_val)

# %% [markdown]
# ### Get Class lables

# %%
# Get Food101 class labels
food101_info = tfds.builder('food101').info
food101_labels = food101_info.features['label'].names

# Combine with Open Images V7 food classes
class_labels = food101_labels + food_classes

# Make sure class_labels is a list and has unique values
class_labels = list(set(class_labels))

# Print the number of classes
print(f"Total number of classes: {len(class_labels)}")

# Update num_classes
num_classes = len(class_labels)

# %% [markdown]
# ### Combine Datasets

# %%
def preprocess_food101(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, tf.cast(label, tf.int32)

def preprocess_open_images(image, label):
    image = tf.ensure_shape(image, (224, 224, 3))
    return image, tf.cast(label, tf.int32)

# Preprocess Food101 dataset
food101_train = food101_train.map(preprocess_food101, num_parallel_calls=tf.data.AUTOTUNE)
food101_val = food101_val.map(preprocess_food101, num_parallel_calls=tf.data.AUTOTUNE)

# Preprocess Open Images dataset
open_v7_train_tf = open_v7_train_tf.map(preprocess_open_images, num_parallel_calls=tf.data.AUTOTUNE)
open_v7_val_tf = open_v7_val_tf.map(preprocess_open_images, num_parallel_calls=tf.data.AUTOTUNE)

# Now combine the datasets
train_dataset = food101_train.concatenate(open_v7_train_tf)
val_dataset = food101_val.concatenate(open_v7_val_tf)

# Then continue with your existing code for preparing the datasets
train_dataset = prepare_dataset(train_dataset)
val_dataset = prepare_dataset(val_dataset)

# Print shapes to verify
for images, labels in train_dataset.take(1):
    print("Train dataset shape:", images.shape, labels.shape)

for images, labels in val_dataset.take(1):
    print("Validation dataset shape:", images.shape, labels.shape)


# %% [markdown]
# ### Model architecture

# %%
num_classes = len(class_labels)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=output)

# Use mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# %% [markdown]
# **Training**

# %%
# Compile model
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras', 
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

callbacks = [
    checkpoint,
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3)
]


# %%
# Training with gradient accumulation
def grad_accumulation_step(model, x, y, optimizer, acc_gradients, num_accumulation_steps):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    for i in range(len(acc_gradients)):
        acc_gradients[i] += gradients[i]
    if acc_gradients[0] is not None:
        optimizer.apply_gradients(zip(acc_gradients, model.trainable_variables))
        for i in range(len(acc_gradients)):
            acc_gradients[i] = tf.zeros_like(acc_gradients[i])
    return loss

# Memory usage logging
def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / 1024 / 1024  # in MB
    print(f"Memory usage: {memory_usage:.2f} MB")

# Start profiler
logdir = './logs'
profiler.start(logdir)

# Initialize gradient accumulation
num_accumulation_steps = 4
acc_gradients = [tf.zeros_like(tv) for tv in model.trainable_variables]

epochs = 10
total_samples = len(food101_train) + len(open_v7_train) 
steps_per_epoch = total_samples // 8  

for epoch in range(epochs):
    for step, (x, y) in enumerate(train_dataset):
        loss = grad_accumulation_step(model, x, y, optimizer, acc_gradients, num_accumulation_steps)
        if step % num_accumulation_steps == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.numpy()}")
            log_memory_usage()
        
        if step >= steps_per_epoch:
            break

    # Validation
    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f"Epoch {epoch}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# Stop profiler
profiler.stop()

# %%
#Fine-tuning
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

total_samples = len(food101_train) + len(open_v7_train) 
steps_per_epoch = total_samples // batch_size
model.fit(train_dataset, epochs=10, steps_per_epoch=steps_per_epoch, validation_data=val_dataset, callbacks=callbacks)

# Save the model
model.save('food_recognition_modelV2.h5')


# %% [markdown]
# **Inference**

# %%
def predict_food(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    return predicted_class

# %% [markdown]
# ### Nutrition data integration

# %%
nutrition_data = pd.read_csv("Nutrition-Data/nutrients_csvfile.csv")

def get_nutrition_info(food_item):
    try:
        nutrition = nutrition_data[nutrition_data['food_item'] == food_item].iloc[0]
        return {
            'calories': nutrition['calories'],
            'protein': nutrition['protein'],
            'carbs': nutrition['carbs'],
            'fat': nutrition['fat']
        }
    except IndexError:
        return get_nutrition_info_from_api(food_item)

# API CALL WHEN NUTRITION VALUERS ARE NOT IN THE LOCAL DATASET
API_KEY = "hydUyBjWVdUlt1qNIeB2dKGgQYbjFiQwMjm6YpBn" 
API_ENDPOINT = "https://api.nal.usda.gov/fdc/v1/foods/search"

def get_nutrition_info_from_api(food_item):
    params = {
        "api_key": API_KEY,
        "query": food_item,
        "dataType": ["Survey (FNDDS)"],
        "pageSize": 1
    }
    
    response = requests.get(API_ENDPOINT, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data['foods']:
            food = data['foods'][0]
            nutrients = food['foodNutrients']
            
            nutrition_info = {
                'calories': next((n['value'] for n in nutrients if n['nutrientName'] == 'Energy'), None),
                'protein': next((n['value'] for n in nutrients if n['nutrientName'] == 'Protein'), None),
                'carbs': next((n['value'] for n in nutrients if n['nutrientName'] == 'Carbohydrate, by difference'), None),
                'fat': next((n['value'] for n in nutrients if n['nutrientName'] == 'Total lipid (fat)'), None)
            }
            
            return nutrition_info
    
    # If API call fails or no data found, return None
    return None

# %% [markdown]
# ### Main function

# %%
def food_recognition_and_nutrition(image_path):
    try:
        predicted_class_index = predict_food(image_path)
        food_item = class_labels[predicted_class_index]
        nutrition_info = get_nutrition_info(food_item)
        
        return {
            'food_item': food_item,
            'nutrition_info': nutrition_info
        }
    except Exception as e:
        logger.error(f"Error in food recognition: {str(e)}")
        return None

# %% [markdown]
# ### TFLITE Model

# %%
# TFLite conversion
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('food_recognition_model_v2.tflite', 'wb') as f:
    f.write(tflite_model)

print("TensorFlow Lite model saved successfully.")


