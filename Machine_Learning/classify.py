
from __future__ import print_function
import os
import numpy as np
import keras
from keras import layers, optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras import backend as K
from keras.layers.core import Lambda

import cv2
from cv2 import imread

def build_model(num_classes):
	# Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
	x_shape = [32,32,3]
	weight_decay = 0.0005
	model = Sequential()
	weight_decay = weight_decay

	model.add(Conv2D(64, (3, 3), padding='same',
						input_shape=x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))

	model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))
	return model

if __name__ == '__main__':
	dataset_path = 'C:\\Users\\kevin\\Desktop\\ECE 196\\DepthMapping\\Machine_Learning\\dataset'
   	# TODO: get labels for each class and the total number 
    classes = [x[0] for x in os.walk(dataset_path)]
    num_classes = len(classes) - 1

    print("number of classes = ", num_classes)    
    for i in classes:
        print(i)
    model_path = './Machine_Learning/personal_train.h5'
   	# TODO: build model and load weights 
    model = build_model(num_classes)
    model.load_weights(model_path)
	
	file_list = []
	for dirpath, dirname, filename in os.walk(path):
		num_classes += 1
		this_label_list = []
		for f in filename:
			fp = os.path.join(dirpath, f)	# image file
			file_list.append(fp)

    # TODO: load data
   	image = cv2.imread(file_list[0])

	if image is None:
		print("Image is of type None")
		continue

	print("File detected!!")

	image = cv2.resize(image, (32,32))
	image = np.expand_dims(image, axis = 0)

	# TODO: classify data
	predicted_values = model.predict(image) # sum of every element adds up to 1
	result = classes[np.argmax(predicted_values, axis = 1)[0] + 1] 

	print(result)
