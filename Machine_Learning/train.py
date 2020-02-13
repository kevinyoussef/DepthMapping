from __future__ import print_function
import os
import numpy as np
import keras
from keras import layers, optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import cv2
from cv2 import imread

DEFAULT_WIDTH = 32
DEFAULT_HEIGHT = 32

'''
Function Name: load_data()
Function Description: this function loads data from a designated dataset directory and returns
	the training set and testing set as numpy array, as well as the number of classes in the dataset
Parameters:
	- path: path of the dataset directory (default = './dataset')
	- max_x: width of the image (default = 32)
	- max_y: height of the image (default = 32)
	- prop: the proportion data reserved for testing set (default = 0.2)
Return Values:
	- x_train: trainind dataset
	- x_test: testing dataset
	- y_train: labels for the training set
	- y_test: labels for the testing set
	- num_classes: number of classes in the dataset
'''


def load_data(path = './dataset/', max_x = 32, max_y = 32, prop = 0.2):
#         print("loading dataset")
        
        x_train = np.empty([0, max_x, max_y, 3])
        x_test = np.empty([0, max_x, max_y, 3])
        
        y_train = np.empty([0])
        y_test = np.empty([0])
        label = -1
        
        for dirpath, dirname, filename in os.walk(path):
                x_data = []
                y_data = []
                for f in filename:
                        fp = os.path.join(dirpath, f)	# image file
                        image = imread(fp)
                        print("loading file: ", fp)
                        image = cv2.resize(image, (max_y,max_x))
                        
                        if len(image.shape) == 3:
                            # image is rgb
                            x_data.append(image)
                            y_data.append(label)
                
                if label != -1:
                    x_data = np.array(x_data)
                    y_data = np.array(y_data)
                    num_of_image = x_data.shape[0]
                    
                    num_of_test = int(num_of_image * prop)
                    num_of_train = num_of_image - num_of_test
                    
                    x_data_train = x_data[0:num_of_train, :]
                    x_data_test = x_data[num_of_train:, :]
                    
                    y_data_train = y_data[0:num_of_train]
                    y_data_test = y_data[num_of_train:]
                    
                    x_train = np.concatenate((x_train, x_data_train), axis = 0)
                    x_test = np.concatenate((x_test, x_data_test), axis = 0)
                    
                    y_train = np.concatenate((y_train, y_data_train), axis = 0)
                    y_test = np.concatenate((y_test, y_data_test), axis = 0)
                    
        
                label += 1
        
        return (x_train, y_train), (x_test, y_test), label
'''

def load_data(path = './Machine_Learning/dataset/', max_x = DEFAULT_WIDTH, max_y = DEFAULT_HEIGHT, prop = 0.2):
	x_train = np.empty([0, max_x, max_y, 3])
	x_test = np.empty([0, max_x, max_y, 3])
	all_data_list = []
	all_labels_list = []
	y_train = np.empty([0])
	y_test = np.empty([0])
	num_classes = -1
	for dirpath, dirname, filename in os.walk(path):
		num_classes += 1
		this_label_list = []
		for f in filename:
			fp = os.path.join(dirpath, f)	# image file
			#print("file: ", fp)
			print(dirpath)
			pic = cv2.imread(fp, 1)
			pic = cv2.resize(pic, (max_x, max_y))
			all_data_list.append(pic)
			name = dirpath.split('/')
			this_label_list.append(num_classes)
		all_labels_list.append(this_label_list)
	all_labels_list.pop(0)
	all_data = np.array(all_data_list)
	test_indeces = []
	shift = 0
	for i in all_labels_list:
		test_indeces.append((np.random.choice(len(i), int(prop*len(i)), replace = False)) + shift)
		shift += len(i)
	flatten = lambda l: [item for sublist in l for item in sublist]
	temp = flatten(test_indeces)
	test_indeces = temp
	all_labels = np.array(flatten(all_labels_list))
	x_test = all_data[test_indeces]
	y_test = all_labels[test_indeces]

	train_indeces = [i for i in range(all_data.shape[0]) if i not in test_indeces]
	
	x_train = all_data[train_indeces]
	y_train = all_labels[train_indeces]

	return (x_train, y_train), (x_test, y_test), num_classes
'''

''' 
Function Name: load_model()
Function Description: this function builds the model 
Parameters:
	- num_classes: number of objects being trained 
Return Value:
	- model: object contraining the model, with weights loaded
'''
def load_model(num_classes):
	model = Sequential()

	# TODO: add a 2D convolution layer with 32 filters, and 6x6 kernal, make this the input layer
	model.add(layers.Conv2D(32, (6,6), input_shape = (DEFAULT_WIDTH, DEFAULT_HEIGHT,1)))
	# TODO: add a relu activation layer
	model.add(layers.Activation('relu'))
	# TODO: add a batch normalization layer
	model.add(layers.BatchNormalization())
	# TODO: add a 2D max pooling layer with 2x2 kernal
	model.add(layers.MaxPool2D((2,2)))
	# TODO: add a flatten layer
	model.add(layers.Flatten())
	# TODO: add a fully-connected layer with 32 units and relu activation function
	model.add(layers.Dense(32, activation='relu'))
	# TODO: add a dropout layer with 30% drop rate
	model.add(layers.Dropout(rate = 0.3))

	model.add(Dense(num_classes, activation = 'softmax'))
	model.summary()

	return model


'''
Function Name: train_model()
Function Description: this function trains the model with hyper-parameters specified by as inputs to the
	function call.
Parameters:
	- model: neural network model created by load_model() function call
	- xTrain: feature vectors for training
	- yTrain: label vectors for training
	- xTest: feature vectors for validation 
	- yTest: label vectors for validation
	- num_classes: num of classes in the dataset (Integer)
	- batchSize: batch size to user per epoch (Integer)
	- max_epoches: number of forward and backword pass through the network
	- learningRage: learning rate used during gradient descent
	- outFile: name of the model to save the weights after training
Return Value:
	- model: trained model
'''
def train_model(model, xTrain, yTrain, xTest, yTest, num_classes, batchSize = 128, max_epoches = 250,learningRate = 0.001, outFile = 'personal_train.h5'):
	
	batch_size = batchSize
	maxepoches = max_epoches
	learning_rate = learningRate
	x = np.arange(5)
	(x_train, y_train), (x_test, y_test) = (xTrain, yTrain),(xTest, yTest)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	print("y train max")
	print(np.max(y_train))   
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
		
	lr_drop = 20

	def lr_scheduler(epoch):
		return learning_rate * (0.5 ** (epoch // lr_drop))
	
	reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)


	#TODO: compile the model with 'categorical_crossentropy' as loss function and
	# stocastic gradient descent optomizer with learning rate specified by 
	# the input parameter and 'accuracy' metrics
	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=True,  # randomly flip images
		vertical_flip=False)  # randomly flip images
	# (std, mean, and principal components if ZCA whitening is applied).
	datagen.fit(x_train)


	
	sgd = optimizers.SGD(lr=learningRate, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

	# TODO: train the model with (x_test, y_test) as validation data, with other hyper-parameters defined
	#			by the inputs to this function call
	print(f'xtrain shape = {x_train.shape}')
	print(f'ytrain shape = {y_train.shape}')

	print(f'datagen.flow shape = {datagen.flow(x_train, y_train, batch_size=batch_size).shape}')

	
	historytemp = model.fit_generator(datagen.flow(x_train, y_train,
									batch_size=batch_size),
							steps_per_epoch=x_train.shape[0] // batch_size,
							epochs=maxepoches, 
							validation_data=(x_test, y_test),
							callbacks=[reduce_lr],
							verbose=1)

	# model.fit(x_train, y_train, batch_size=batchSize, epochs=max_epoches, validation_data=(x_test, y_test))

	# TODO: save model weight to the file specified by the 'outFile' parameter

	model.save_weights(outFile)
	return model


if __name__ == '__main__':

	dataset_path = './dataset/' 

	num_classes = 0
	x_train = []
	x_test = []
	y_train = []
	y_test = []
	(x_train, y_train), (x_test, y_test), num_classes = load_data(path = dataset_path)

	# TODO: remove exit(-1) when load_data() is completed


	# TODO: remove exit(-1) once load_model() is completed
	model = load_model(num_classes) 

	# TODO: remove exit(-1) once train_model() is completed
	model = train_model(model, x_train, y_train, x_test, y_test, num_classes)

	predicted_x = model.predict(x_test)
	residuals = np.argmax(predicted_x,1)==y_test

	loss = sum(residuals)/len(residuals)
	print("The validation 0/1 loss is: ",loss)  






