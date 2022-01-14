import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

#loading MNIST dataset
from keras.datasets import mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

#Network Architecture
model = keras.models.Sequential([
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

#Compilation Step
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Preparing Data
train_images=train_images.reshape((60000, 28*28))
train_images=train_images.astype('float32')/255
test_images=test_images.reshape((10000, 28*28))
test_images=test_images.astype('float32')/255

#'Fitting the model' or Training the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print(predictions[0].argmax(),"\n",test_labels[0])
