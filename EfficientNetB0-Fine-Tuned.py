import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Load the Food-101 dataset
food101_train, food101_test = tfds.load('food101', split=['train', 'validation'], as_supervised=True)

# Image dimensions
img_height, img_width = 224, 224
batch_size = 32

# Preprocessing function for the dataset
def preprocess(image, label):
    image = tf.image.resize(image, (img_height, img_width))
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    return image, label

# Apply preprocessing to the datasets
train_dataset = food101_train.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
validation_dataset = food101_test.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Load the EfficientNetB0 model without the top layers
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model
base_model.trainable = False

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(101, activation='softmax')(x)  # 101 classes in Food-101

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

# Unfreeze some layers of the base model for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Recompile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Continue training the model
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

# Save the model
model.save('fine_tuned_efficientnetb0_food101.h5')
