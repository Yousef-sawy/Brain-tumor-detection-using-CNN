#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/Users/reem/Desktop/MLlab/Project/Brain Tumor'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[26]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score


# In[27]:


import ipywidgets as widgets
import io
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import tensorflow as tf


# Paths for folder
# 

# In[28]:


X_train = []
Y_train = []
image_size = 150
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
for i in labels:
    folderPath = os.path.join('/Users/reem/Desktop/MLlab/Project/Brain Tumor/Training',i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)
        
for i in labels:
    folderPath = os.path.join('/Users/reem/Desktop/MLlab/Project/Brain Tumor/Testing',i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)

X_train = np.array(X_train)
Y_train = np.array(Y_train)


# In[29]:


X_train,Y_train = shuffle(X_train,Y_train,random_state=101)
X_train.shape


# Train test split

# In[30]:


X_train,X_test,y_train,y_test = train_test_split(X_train,Y_train,test_size=0.1,random_state=101)


# In[31]:


y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train=y_train_new
y_train = tf.keras.utils.to_categorical(y_train)

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test=y_test_new
y_test = tf.keras.utils.to_categorical(y_test)


# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from tqdm import tqdm
from warnings import filterwarnings
import pickle
import os
import shutil
import random
import cv2

#Import sklearn modules
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, average_precision_score

#Import tf modules
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import InceptionResNetV2

filterwarnings("ignore")
np.random.seed(0)


# In[2]:


train_img = []
train_labels_not_encoded = []

test_img = []
test_labels_not_encoded = []

path_train = ('/Users/reem/Desktop/MLlab/Project/Brain Tumor/Training/')
path_test = ('/Users/reem/Desktop/MLlab/Project/Brain Tumor/Testing/')

img_size= 180
for i in os.listdir(path_train):
    for j in tqdm(os.listdir(path_train+i)):
        train_img.append(cv2.resize(cv2.imread(path_train+i+'/'+j), (img_size,img_size))) 
        train_labels_not_encoded.append(i)
        
for i in os.listdir(path_test):
    for j in tqdm(os.listdir(path_test+i)):
        test_img.append(cv2.resize(cv2.imread(path_test+i+'/'+j), (img_size,img_size))) 
        test_labels_not_encoded.append(i)
        
train_img = np.array(train_img)
test_img = np.array(test_img)


# In[3]:


y_train = pd.get_dummies(train_labels_not_encoded)
y_test = pd.get_dummies(test_labels_not_encoded)
labels=y_train.columns


# In[4]:


fig, ax = plt.subplots(1,4,figsize=(20,20))
fig.text(s='Sample Image From Each Label',size=18,fontweight='bold',
         color="black",y=0.61,x=0.4,alpha=0.9)
for count, ele in enumerate(os.listdir(path_train)):
    for j in os.listdir(path_train+ele):
        img= cv2.imread(path_train+ele+'/'+j)
        ax[count].imshow(img)
        ax[count].set_title(ele)
        ax[count].axis('off')
        break


# In[5]:


print("Train size:", train_img.shape[0], "Test size:", test_img.shape[0])


# In[6]:


y_train


# # DNN

# In[9]:


import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


# Load the data
X_train = []
Y_train = []
image_size = 150
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


# In[11]:


# Load training data
for i in labels:
    folderPath = os.path.join('/Users/reem/Desktop/MLlab/Project/Brain Tumor/Training',i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)


# In[12]:


# Load testing data
for i in labels:
    folderPath = os.path.join('/Users/reem/Desktop/MLlab/Project/Brain Tumor/Testing', i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        Y_train.append(i)

X_train = np.array(X_train)
Y_train = np.array(Y_train)


# In[13]:


# Shuffle the data
X_train, Y_train = shuffle(X_train, Y_train, random_state=101)


# In[14]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=101)


# In[15]:


# Convert labels to categorical
y_train_new = [labels.index(i) for i in y_train]
y_train = tf.keras.utils.to_categorical(y_train_new)

y_test_new = [labels.index(i) for i in y_test]
y_test = tf.keras.utils.to_categorical(y_test_new)


# In[18]:


# Create the Dense Neural Network (DNN) model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(150, 150, 3)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.summary()


# In[19]:


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[20]:


# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_split=0.1)


# In[22]:


# Plot the training and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.legend(loc='upper left')
plt.show()


# In[23]:


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.legend(loc='upper left')
plt.show()


# In[25]:


# Evaluate the model on the test data
Y_pred = model.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(Y_pred, axis=1)


# In[26]:


# Calculate and print the accuracy score
accuracy = accuracy_score(y_test_labels, y_pred_labels)
print("Test Accuracy:", accuracy)


# In[27]:


# Calculate and print the confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels)
print("Confusion Matrix:")
print(cm)


# In[28]:


# Print the classification report
report = classification_report(y_test_labels, y_pred_labels)
print("Classification Report:")
print(report)


# # VGG16

# In[29]:


import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import accuracy_score

# Set the paths to the training and testing datasets
train_dir = '/Users/reem/Desktop/MLlab/Project/Brain Tumor/Training'
test_dir = '/Users/reem/Desktop/MLlab/Project/Brain Tumor/Testing'

# Set the image size and batch size
image_size = (150, 150)
batch_size = 32

# Create data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load the pre-trained VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

# Freeze the weights of the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Create the model architecture by adding fully connected layers on top of the base model
model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

# Predict on the test set
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Get the true labels from the test set
y_true = test_generator.classes

# Calculate the accuracy score
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)


# In[5]:


import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set the paths to the training and testing datasets
train_dir = '/Users/reem/Desktop/MLlab/Project/Brain Tumor/Training'
test_dir = '/Users/reem/Desktop/MLlab/Project/Brain Tumor/Testing'

# Set the image size and batch size
image_size = (224, 224)
batch_size = 32

# Create data generators for training and testing with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load the pre-trained VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

# Freeze the weights of the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Create the model architecture by adding fully connected layers on top of the base model
model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

# Evaluate the model
model_evaluation = model.evaluate(test_generator)
print("Test Loss:", model_evaluation[0])
print("Test Accuracy:", model_evaluation[1])

# Generate predictions on the test set
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

# Generate classification report
print(classification_report(y_true, y_pred))

# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss")
ax[0].legend(loc='best')
ax[0].set_title('Loss')

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
ax[1].legend(loc='best')
ax[1].set_title('Accuracy')

plt.tight_layout()
plt.show()


# In[ ]:





# # ResNet50

# In[1]:


import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.metrics import classification_report

# Set the paths to the training and testing datasets
train_dir = '/Users/reem/Desktop/MLlab/Project/Brain Tumor/Training'
test_dir = '/Users/reem/Desktop/MLlab/Project/Brain Tumor/Testing'

# Set the image size and batch size
image_size = (150, 150)
batch_size = 32

# Create data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load the pre-trained ResNet50 model without the top (fully connected) layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

# Freeze the weights of the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Create the model architecture by adding fully connected layers on top of the base model
model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

# Predict on the test set
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Get the true labels from the test set
y_true = test_generator.classes

# Calculate evaluation metrics
report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys())
print("Classification Report:")
print(report)


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot the accuracy and loss curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
ax1.plot(epochs, acc, 'r', label="Training Accuracy")
ax1.plot(epochs, val_acc, 'b', label="Validation Accuracy")
ax1.set_title('Training and Validation Accuracy')
ax1.legend(loc='upper left')

ax2.plot(epochs, loss, 'r', label="Training Loss")
ax2.plot(epochs, val_loss, 'b', label="Validation Loss")
ax2.set_title('Training and Validation Loss')
ax2.legend(loc='upper left')

plt.show()


# # Other Classifiers

# In[6]:


import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Set the paths to the training and testing datasets
train_dir = '/Users/reem/Desktop/MLlab/Project/Brain Tumor/Training'
test_dir = '/Users/reem/Desktop/MLlab/Project/Brain Tumor/Testing'

# Set the image size and batch size
image_size = (150, 150)
batch_size = 32

# Create data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Load the training data and labels
X_train = []
y_train = []
for i in range(len(train_generator)):
    images, labels = train_generator[i]
    X_train.extend(images)
    y_train.extend(np.argmax(labels, axis=1))

# Load the testing data and labels
X_test = []
y_test = []
for i in range(len(test_generator)):
    images, labels = test_generator[i]
    X_test.extend(images)
    y_test.extend(np.argmax(labels, axis=1))

# Convert the data to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Reshape the data if necessary (e.g., for KNN)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Create and train the Naive Bayes classifier
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# Predict on the test set using the KNN classifier
y_pred_knn = knn.predict(X_test)

# Predict on the test set using the Naive Bayes classifier
y_pred_nb = naive_bayes.predict(X_test)

# Get the target names for the classification report
target_names = list(test_generator.class_indices.keys())

# Print the classification report for the KNN classifier
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn, target_names=target_names))

# Print the classification report for the Naive Bayes classifier
print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb, target_names=target_names))


# # Convolutional Neural Network

# In[34]:


model = Sequential()
model.add(Conv2D(32,(3,3),activation = 'relu',input_shape=(150,150,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512,activation = 'relu'))
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(4,activation='softmax'))


# In[35]:


model.summary()


# In[36]:


model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[37]:


history = model.fit(X_train,y_train,epochs=20,validation_split=0.1)


# In[55]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[56]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))
fig = plt.figure(figsize=(14,7))
plt.plot(epochs,acc,'r',label="Training Accuracy")
plt.plot(epochs,val_acc,'b',label="Validation Accuracy")
plt.legend(loc='upper left')
plt.show()


# In[57]:


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
fig = plt.figure(figsize=(14,7))
plt.plot(epochs,loss,'r',label="Training loss")
plt.plot(epochs,val_loss,'b',label="Validation loss")
plt.legend(loc='upper left')
plt.show()


# Predicting Brain Tumor

# In[58]:


img = cv2.imread('/Users/reem/Desktop/MLlab/Project/Brain Tumor/Training/pituitary_tumor/p (107).jpg')
img = cv2.resize(img,(150,150))
img_array = np.array(img)
img_array.shape


# In[59]:


img_array = img_array.reshape(1,150,150,3)
img_array.shape


# In[60]:


from tensorflow.keras.preprocessing import image
img = image.load_img('/Users/reem/Desktop/MLlab/Project/Brain Tumor/Training/pituitary_tumor/p (107).jpg')
plt.imshow(img,interpolation='nearest')
plt.show()


# In[61]:


a=model.predict(img_array)
indices = a.argmax()
indices
print(labels[indices])


# In[62]:


from tensorflow.keras.preprocessing import image
img = image.load_img('/Users/reem/Desktop/MLlab/Project/Brain Tumor/Training/no_tumor/1.jpg')
plt.imshow(img,interpolation='nearest')
plt.show()


# In[63]:


a=model.predict(img_array)
indices = a.argmax()
indices
print(labels[indices])


# In[ ]:


from tensorflow.keras.preprocessing import image
img = image.load_img('/Users/reem/Desktop/MLlab/Project/Brain Tumor/Training/meningioma_tumor/m (2).jpg')
plt.imshow(img,interpolation='nearest')
plt.show()


# In[ ]:


a=model.predict(img_array)
indices = a.argmax()
indices
print(labels[indices])


# In[ ]:


from tensorflow.keras.preprocessing import image
img = image.load_img('/Users/reem/Desktop/MLlab/Project/Brain Tumor/Training/glioma_tumor/gg (1).jpg')
plt.imshow(img,interpolation='nearest')
plt.show()


# In[ ]:


a=model.predict(img_array)
indices = a.argmax()
indices
print(labels[indices])


# In[65]:


from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load and display the image
img_path1 = '/Users/reem/Desktop/MLlab/Project/Brain Tumor/Training/no_tumor/1.jpg'
img1 = image.load_img(img_path1)
plt.imshow(img1, interpolation='nearest')
plt.show()

# Preprocess the image
img1 = img1.resize((150, 150))
img_array1 = image.img_to_array(img1)
img_array1 = img_array1 / 255.0  # Normalize the pixel values
img_array1 = img_array1.reshape(1, 150, 150, 3)

# Make prediction
prediction1 = model.predict(img_array1)
predicted_label1 = labels[np.argmax(prediction1)]
print("Predicted label:", predicted_label1)


# Accuracy Score

# In[ ]:


Y_pred=model.predict(X_test)
Y_pred


# In[ ]:


loss, acc = model.evaluate(x=X_test, y=y_test)


# In[ ]:


print (f"Test Loss = {loss}")
print (f"Test Accuracy = {acc}")


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# Convert `y_test` and `Y_pred` to the appropriate format if necessary
y_test = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
Y_pred = np.argmax(Y_pred, axis=1) if len(Y_pred.shape) > 1 else Y_pred

# Calculate and print the confusion matrix
print('Confusion Matrix')
cm = confusion_matrix(y_test, Y_pred)
print(cm)


# In[ ]:


print(classification_report(y_test,Y_pred))

