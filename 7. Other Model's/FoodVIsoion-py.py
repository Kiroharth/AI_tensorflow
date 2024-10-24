# %% [markdown]
# <a href="https://colab.research.google.com/github/gauravreddy08/food-vision/blob/main/model_training.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # **Food Vision** 🍔 
# 
# As an introductory project to myself, I built an **end-to-end CNN Image Classification Model** which identifies the food in your image.
# 
# I worked out with a pretrained Image Classification Model that comes with Keras and then retrained it on the infamous **Food101 Dataset**.
# 
# 
# **Fun Fact :**
# 
# The Model actually beats the DeepFood Paper's model which also trained on the same dataset.
#  
# The Accuracy of [**DeepFood**](https://arxiv.org/abs/1606.05675) was **77.4%** and our model's is **85%**. Difference of **8%** ain't much but the interesting thing is, DeepFood's model took 2-3 days to train while our's was around 60min.
# 
# > **Dataset :** `Food101`
# 
# > **Model :** `EfficientNetB1`
# 
# 

# %% [markdown]
# 
# 
# ## **Setting up the Workspace**
# 
# * Checking the GPU
# * Mounting Google Drive
# * Importing Tensorflow
# * Importing other required Packages
# 
# ### **Checking the GPU**
# 
# For this Project we will working with **Mixed Precision**. And mixed precision works best with a with a GPU with compatibility capacity **7.0+**.
# 
# At the time of writing, colab offers the following GPU's :
# * Nvidia K80
# * **Nvidia T4**
# * Nvidia P100
# 
# Colab allocates a random GPU everytime we factory reset runtime. So you can reset the runtime till you get a **Tesla T4 GPU** as T4 GPU has a rating 7.5.
# 
# > In case using local hardware, use a GPU with rating 7.0+ for better results.
# 
# Run the below cell to see which GPU is allocated to you.

# %%
!nvidia-smi -L

# %% [markdown]
# 
# ### **Mounting Google Drive**
# 
# 
# 

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ### **Importing Tensorflow**
# 
# At the time of writing, `tesnorflow 2.5.0` has a bug with EfficientNet Models. [Click Here](https://github.com/tensorflow/tensorflow/issues/49725) to get more info about the bug. Hopefully tensorflow fixes it soon.
# 
# So the below code is used to downgrade the version to `tensorflow 2.4.1`, it will take a moment to uninstall the previous version and install our required version.
# 
# > You need to restart the **Runtime** after required version of tensorflow is installed. 
# 
# **Note :** Restarting runtime won't assign you a new GPU.

# %%
!pip install tensorflow==2.4.1
import tensorflow as tf
print(tf.__version__)

# %% [markdown]
# ### **Importing other required Packages**

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import tensorflow_datasets as tfds
import seaborn as sn

# %% [markdown]
# #### **Importing `helper_fuctions`**
# 
# The `helper_functions.py` is a python script created by me. Which has some important functions I use frequently while building Deep Learning Models.

# %%
!wget https://raw.githubusercontent.com/gauravreddy08/deep-learning-tensorflow/main/extras/helper_function.py

# %%
from helper_function import plot_loss_curves, load_and_prep_image

# %% [markdown]
# ## **Getting the Data Ready**
# 
# The Dataset used is **Food101**, which is available on both Kaggle and Tensorflow. 
# 
# In the below cells we will be importing Datasets from `Tensorflow Datasets` Module.
# 

# %%
# Prints list of Datasets avaible in Tensorflow Datasets Module

dataset_list = tfds.list_builders()
dataset_list[:10]

# %% [markdown]
# ### **Importing Food101 Dataset**
# 
# **Disclaimer :** 
# The below cell will take time to run, as it will be downloading 
# **4.65GB data** from **Tensorflow Datasets Module**. 
# 
# So do check if you have enough **Disk Space** and **Bandwidth Cap** to run the below cell.

# %%
(train_data, test_data), ds_info = tfds.load(name='food101',
                                             split=['train', 'validation'],
                                             shuffle_files=False,
                                             as_supervised=True,
                                             with_info=True)

# %% [markdown]
# ## **Becoming One with the Data**
# 
# One of the most important steps in building any ML or DL Model is to **become one with the data**. 
# 
# Once you get the gist of what type of data your dealing with and how it is structured, everything else will fall in place.

# %%
ds_info.features

# %%
class_names = ds_info.features['label'].names
class_names[:10]

# %%
train_one_sample = train_data.take(1)

# %%
train_one_sample

# %%
for image, label in train_one_sample:
  print(f"""
  Image Shape : {image.shape}
  Image Datatype : {image.dtype}
  Class : {class_names[label.numpy()]}
  """)

# %%
image[:2]

# %%
tf.reduce_min(image), tf.reduce_max(image)

# %%
plt.imshow(image)
plt.title(class_names[label.numpy()])
plt.axis(False);

# %% [markdown]
# ## **Preprocessing the Data**
# 
# Since we've downloaded the data from TensorFlow Datasets, there are a couple of preprocessing steps we have to take before it's ready to model. 
# 
# More specifically, our data is currently:
# 
# * In `uint8` data type
# * Comprised of all differnet sized tensors (different sized images)
# * Not scaled (the pixel values are between 0 & 255)
# 
# Whereas, models like data to be:
# 
# * In `float32` data type
# * Have all of the same size tensors (batches require all tensors have the same shape, e.g. `(224, 224, 3)`)
# * Scaled (values between 0 & 1), also called normalized
# 
# To take care of these, we'll create a `preprocess_img()` function which:
# 
# * Resizes an input image tensor to a specified size using [`tf.image.resize()`](https://www.tensorflow.org/api_docs/python/tf/image/resize)
# * Converts an input image tensor's current datatype to `tf.float32` using [`tf.cast()`](https://www.tensorflow.org/api_docs/python/tf/cast)

# %%
def preprocess_img(image, label, img_size=224):
  image = tf.image.resize(image, [img_size, img_size])
  image = tf.cast(image, tf.float16)
  return image, label

# %%
# Trying the preprocess function on a single image

preprocessed_img = preprocess_img(image, label)[0]
preprocessed_img

# %%
train_data = train_data.map(preprocess_img, tf.data.AUTOTUNE)
train_data = train_data.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.AUTOTUNE)

test_data = test_data.map(preprocess_img, tf.data.AUTOTUNE)
test_data = test_data.batch(32)

# %%
train_data

# %%
test_data

# %% [markdown]
# ## **Building the Model : EfficientNetB1**
# 
# 
# ### **Getting the Callbacks ready**
# As we are dealing with a complex Neural Network (EfficientNetB0) its a good practice to have few call backs set up. Few callbacks I will be using throughtout this Notebook are :
#  * **TensorBoard Callback :** TensorBoard provides the visualization and tooling needed for machine learning experimentation
# 
#  * **EarlyStoppingCallback :** Used to stop training when a monitored metric has stopped improving.
#  
#  * **ReduceLROnPlateau :** Reduce learning rate when a metric has stopped improving.
# 
# 
#  We already have **TensorBoardCallBack** function setup in out helper function, all we have to do is get other callbacks ready.

# %%
from helper_function import create_tensorboard_callback

# %%
# EarlyStopping Callback

early_stopping_callback = tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=3, verbose=1, monitor="val_accuracy")

# %%
# ReduceLROnPlateau Callback

lower_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2,
                                                monitor='val_accuracy',
                                                min_lr=1e-7,
                                                patience=0,
                                                verbose=1)

# %% [markdown]
# 
# 
# ### **Mixed Precision Training**
# Mixed precision is used for training neural networks, reducing training time and memory requirements without affecting the model performance.
# 
# More Specifically, in **Mixed Precision** we will setting global dtype as `mixed_float16`. Because modern accelerators can run operations faster in the 16-bit dtypes, as they have specialized hardware to run 16-bit computations and 16-bit dtypes can be read from memory faster.
# 
# To know more about Mixed Precision, [**click here**](https://www.tensorflow.org/guide/mixed_precision)

# %%
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy(policy='mixed_float16')

# %%
mixed_precision.global_policy()

# %% [markdown]
# 
# 
# ### **Building the Model**

# %%
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Create base model
input_shape = (224, 224, 3)
base_model = tf.keras.applications.EfficientNetB1(include_top=False)

# Input and Data Augmentation
inputs = layers.Input(shape=input_shape, name="input_layer")
x = base_model(inputs)

x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
x = layers.Dropout(.3)(x)

x = layers.Dense(len(class_names))(x)
outputs = layers.Activation("softmax")(x)
model = tf.keras.Model(inputs, outputs)

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy", 
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=["accuracy"])

# %%
model.summary()

# %%
history = model.fit(train_data,
                    epochs=50,
                    steps_per_epoch=len(train_data),
                    validation_data=test_data,
                    validation_steps=int(0.15 * len(test_data)),
                    callbacks=[create_tensorboard_callback("training-logs", "EfficientNetB1-"),
                               early_stopping_callback,
                               lower_lr])

# %%
# Saving the model
model.save("/content/drive/My Drive/FinalModel.hdf5")

# %%
# Saving the model
model.save("FoodVision.hdf5")

# %%
plot_loss_curves(history)

# %%
model.evaluate(test_data)

# %% [markdown]
# ## **Evaluating our Model**

# %%
%load_ext tensorboard
%tensorboard --logdir training-logs

# %%
pred_probs = model.predict(test_data, verbose=1)
len(pred_probs), pred_probs.shape

# %%
pred_classes = pred_probs.argmax(axis=1)
pred_classes[:10], len(pred_classes), pred_classes.shape

# %%
# Getting true labels for the test_data

y_labels = []
test_images = []
for images, labels in test_data.unbatch():
  y_labels.append(labels.numpy())
y_labels[:10]

# %%
# Predicted Labels vs. True Labels
pred_classes==y_labels

# %% [markdown]
# ### **Sklearn's Accuracy Score**

# %%
from sklearn.metrics import accuracy_score

sklearn_acc = accuracy_score(y_labels, pred_classes)
sklearn_acc 

# %% [markdown]
# ### **Confusion Matrix**
# A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known

# %%
cm = tf.math.confusion_matrix(y_labels, pred_classes)

plt.figure(figsize = (200, 200));
sn.heatmap(cm, annot=True, 
           fmt='',
           cmap='Blues');

# %% [markdown]
# ### **Model's Class-wise Accuracy Score**

# %%
from sklearn.metrics import classification_report
report = (classification_report(y_labels, pred_classes, output_dict=True))

# %%
# Create empty dictionary
class_f1_scores = {}
# Loop through classification report items
for k, v in report.items():
  if k == "accuracy": # stop once we get to accuracy key
    break
  else:
    # Append class names and f1-scores to new dictionary
    class_f1_scores[class_names[int(k)]] = v["f1-score"]
class_f1_scores

# %%
report_df = pd.DataFrame(class_f1_scores, index = ['f1-scores']).T

# %%
report_df = report_df.sort_values("f1-scores", ascending=True)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 25))
scores = ax.barh(range(len(report_df)), report_df["f1-scores"].values)
ax.set_yticks(range(len(report_df)))
plt.axvline(x=0.85, linestyle='--', color='r')
ax.set_yticklabels(class_names)
ax.set_xlabel("f1-score")
ax.set_title("F1-Scores for 10 Different Classes")
ax.invert_yaxis(); # reverse the order

# %% [markdown]
# ### **Predicting on our own Custom images**
# 
# Once we have our model ready, its cruicial to evaluate it on our custom data : the data our model has never seen.
# 
# Training and evaluating a model on train and test data is cool, but making predictions on our own realtime images is another level.
# 
# 

# %%
# Get custom food images filepaths
import os

custom_food_images = ["/content/drive/MyDrive/FoodVisionModels/Custom Images/" + img_path for img_path in os.listdir("/content/drive/MyDrive/FoodVisionModels/Custom Images")]
custom_food_images

# %%
def pred_plot_custom(folder_path):
  import os

  custom_food_images = [folder_path + img_path for img_path in os.listdir(folder_path)]
  i=0
  fig,a =  plt.subplots(len(custom_food_images),2, figsize=(15, 5*len(custom_food_images)))

  for img in custom_food_images:
    img = load_and_prep_image(img, scale=False) 
    pred_prob = model.predict(tf.expand_dims(img, axis=0)) 
    pred_class = class_names[pred_prob.argmax()]
    top_5_i = (pred_prob.argsort())[0][-5:][::-1]
    values = pred_prob[0][top_5_i] 
    labels = []
    for x in range(5):
      labels.append(class_names[top_5_i[x]])

    # Plotting Image
    a[i][0].imshow(img/255.) 
    a[i][0].set_title(f"Prediction: {pred_class}   Probability: {pred_prob.max():.2f}")
    a[i][0].axis(False)

    # Plotting Models Top 5 Predictions
    a[i][1].bar(labels, values, color='orange');
    a[i][1].set_title('Top 5 Predictions')
    i=i+1

# %%
pred_plot_custom("/content/drive/MyDrive/FoodVisionModels/Custom Images/")


