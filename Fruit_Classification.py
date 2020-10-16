# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:33:45 2020

@author: user
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D

#initialize CNN
classifier = Sequential()
#adding convolution layers and pooling layers
classifier.add(Convolution2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32, (3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#flattening the layer
classifier.add(Flatten())
#add fully connected layers
classifier.add(Dense(units=32, activation='relu'))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=256, activation='relu'))
classifier.add(Dense(units=6, activation='softmax'))

#compiling CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#fitting CNN to images
from keras.preprocessing.image import ImageDataGenerator
train_data = ImageDataGenerator(rescale=1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_data = ImageDataGenerator(rescale=1./255)

#get training and testing images
training_dataset = train_data.flow_from_directory('train', target_size = (64,64), batch_size = 12, class_mode = 'categorical')
testing_dataset = test_data.flow_from_directory('test', target_size = (64,64), batch_size = 12, class_mode = 'categorical')

classifier.fit_generator(training_dataset, samples_per_epoch = 1212, nb_epoch = 30, validation_data = testing_dataset, nb_val_samples = 300)