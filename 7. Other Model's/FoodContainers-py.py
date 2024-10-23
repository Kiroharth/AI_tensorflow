# %%
import tensorflow as tf
from keras import datasets,layers, models
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# %%
# ================ DATA PREPARATION STAGE ========================== #
# load testing and training data into coarse and fine label respectively
(xcoarse_train, ycoarse_train), (xcoarse_test, ycoarse_test) = datasets.cifar100.load_data(label_mode='coarse')
print('Coarse Class: {}' .format(np.unique(ycoarse_train)))

(xfine_train, yfine_train), (xfine_test, yfine_test) = datasets.cifar100.load_data(label_mode='fine')
print('Fine Class for all: {}' .format(np.unique(yfine_train)))

# %%
# aim for index 3 for coarse, 16-20 for fine for group assignment. (coarse: food containers, fine: bowls,cans,cups, etc..)
idx = [i for i in range(len(ycoarse_train)) if ycoarse_train[i] == 3]
# checks the coarse label of each sample in the training dataset
# append the index of an input image with "Food containers" coarse label.    

print('Total images with 3 coarse label (Food Containers) from TRAINING DATASET: {}' .format(len(idx)))
idx = np.array(idx)

# %%
# Extract all image and corresponding "fine" label and store in train_images, train_labels variable list.
train_images, train_labels = xfine_train[idx], yfine_train[idx]
print("Shape of the image training dataset: {}".format(train_images.shape))
uniq_fineClass = np.unique(train_labels)
print('Fine Class for the extracted training images: {}'.format(uniq_fineClass))

# %%
idx = [i for i in range(len(ycoarse_test)) if ycoarse_test[i] == 3]
# checks the coarse label of each sample in the testing dataset
# append the index of an input image with "Food containers" coarse label.    

print('Total images with 3 coarse label (Food Containers) from TESTING DATASET: {}' .format(len(idx)))
idx = np.array(idx)

# %%
# Extract all image and corresponding "fine" label and store in test_images, test_labels variable list.
test_images, test_labels = xfine_test[idx], yfine_test[idx]
print("Shape of the image testing dataset: {}".format(test_images.shape))
uniq_fineClass = np.unique(test_labels)
print('Fine Class for the extracted testing images: {}'.format(uniq_fineClass))

# %%
# Relabel training and testing dataset to start from zero (0).
for i in range(len(uniq_fineClass)):
  for j in range(len(train_labels)):
    if train_labels[j] == uniq_fineClass[i]:
      train_labels[j] = i

  for j in range(len(test_labels)):
    if test_labels[j] == uniq_fineClass[i]:
      test_labels[j] = i

# %%
# Plot few samples from images from the TESTING DATASET
plt.figure(figsize=(10,2))
for i in range(10):
  plt.subplot(1,10,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(test_images[i], cmap=plt.cm.binary)
  plt.xlabel(test_labels[i])

# %% [markdown]
# 

# %%
# Build the model
model = tf.keras.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))  

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))  

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))  

model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))  

model.add(Dense(len(uniq_fineClass), activation='softmax'))

model.summary()

# Use Adam optimizer with a lower learning rate
optimizer = Adam(learning_rate=0.0005)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)

metricInfo = model.fit(train_images, train_labels, epochs=1000, validation_split=0.1, callbacks=[early_stopping]) #callbacks=[early_stopping]

loss = metricInfo.history['loss']
val_loss = metricInfo.history['val_loss']
epochs = range(1, len(loss) + 1)

# Define a callback to save the best model during training
checkpoint_filepath = 'C:\\Users\\Zamskie\\Documents\\jason 3rd year\\2ND SEM\\cpe emerging\\bestModel\\best_model2.keras'
model_checkpoint = ModelCheckpoint(
    checkpoint_filepath,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train the model with the ModelCheckpoint callback
history = model.fit(
    train_images,
    train_labels,
    epochs=1000,  # Adjust the number of epochs
    validation_split=0.1,
    callbacks=[model_checkpoint]
)

# Load the best model
#best_model = tf.keras.models.load_model(checkpoint_filepath)

# Evaluate the model or use it for predictions
#test_loss, test_accuracy = best_model.evaluate(test_images, test_labels)
#print('Test Accuracy of the Best Model:', test_accuracy)

plt.clf()
plt.plot(epochs, loss, 'g-', label="Training loss")
plt.plot(epochs, val_loss, 'b-', label='Validation loss')
plt.title('Training vs Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
# Test the model
str_class = ['bottles', 'bowl', 'cans', 'cups', 'plates']

print(test_images.shape)
print("Class in the testing image: {}".format(np.unique(test_labels)))
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Total number of testing image: {}'.format(len(test_images)))
print('Test accuracy:', test_acc)

# Another way to test using "prediction" method
classification = model.predict(test_images)
print('\nDisplaying prediction of the first test input image: {}'.format(classification[0]))

# get the index of the maximum probability in the classification[0] result
max_prob_idx = np.argmax(classification[0])
print('Predicted class: {}--{}'.format(max_prob_idx, str_class[max_prob_idx]))
idx = test_labels[0]
print('True class: {} -- {}'.format(idx[0], str_class[idx[0]]))

# Evaluate the model on a per-class basis
y_true = test_labels
y_pred = np.argmax(classification, axis=1)

report = classification_report(y_true, y_pred, target_names=str_class)
print('\nClassification Report:\n', report)


