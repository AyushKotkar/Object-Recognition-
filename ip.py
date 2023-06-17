import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img


#print(os.getcwd())
room_types = os.listdir('dataset')
print(room_types)

print("Types of room found: ", len(room_types))

rooms = []

rooms = []

for item in room_types:
    all_rooms = os.listdir('dataset' + '/' + item)

    for room in all_rooms:
        rooms.append((item, str('dataset' + '/' + item) + '/' + room))

#print(rooms[:1])


rooms_df = pd.DataFrame(data = rooms, columns=['room type', 'image']) 
#print(rooms_df.head())
#print(rooms_df.tail())

#print("Total no. of rooms in the dataset" ,len(rooms_df))

rooms_count = rooms_df['room type'].value_counts()
#print("Rooms in each category: ", rooms_count)

path = 'dataset/'

im_size = 300

images = []
labels = []

for i in room_types:
    data_path = path + str(i)
    filenames = [i for i in os.listdir(data_path)]

    for f in filenames:
        img = cv2.imread(data_path + '/' + f)
        img = cv2.resize(img, (im_size,im_size))
        images.append(img)
        labels.append(i)

images = np.array(images)
#print(images.shape)

images = images.astype('float32') / 255.0
#print(images.shape)

y = rooms_df['room type'].values
#print(y[:5])

y_labelencoder = LabelEncoder()
y = y_labelencoder.fit_transform(y)
#print(y)

images, y = shuffle(images, y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(images, y, test_size=0.05, random_state = 415)

#print(train_x.shape)
#print(train_y.shape)
#print(test_x.shape)
#print(test_y.shape)

#logits = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(300,300,3)),
    keras.layers.Dense(256,activation=tf.nn.tanh),

    keras.layers.Dense(3,activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x,train_y,epochs=5)

#test_x, test_y = load_data('test')
y_pred_prob = model.predict(test_x)
y_pred = np.argmax(y_pred_prob, axis=1)

image = load_img('t2.jpg', target_size = (300,300))
image

image =  np.array(image)
image.shape
image = image.reshape((1,image.shape[0], image.shape[1], image.shape[2]))

yhat = model.predict(image)
print(yhat)