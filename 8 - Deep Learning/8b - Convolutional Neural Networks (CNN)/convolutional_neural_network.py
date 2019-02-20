# Image Detection(Cats & Dogs) : Convolutional Neural Network

# Installing Theano
# conda install theano

# Installing Keras
# pip install keras

# Configure Keras to use Theno backend
# Update ~/.keras/keras.json to change backend to "theano" from "tensorflow"

# Dataset link:
# https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/14_page_p8s40_file_1.zip

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Step 1: Convolution
classifier.add(Convolution2D(filters = 32,
                             kernel_size = [3,3],
                             input_shape = (64, 64, 3),
                             data_format = 'channels_last',
                             activation = 'relu'))

# Step 2: Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding a second convolution layer
classifier.add(Convolution2D(filters = 32,
                             kernel_size = [3,3],
                             activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3: Flattening
classifier.add(Flatten())

# Step 4: Full Connection 
# Adding hidden layer
classifier.add(Dense(units = 128,
                     activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units = 1,
                     activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

#  Part 2 - Fitting the CNN to the images 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64,64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)
