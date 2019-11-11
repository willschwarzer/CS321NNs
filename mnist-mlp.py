import os
import os.path
import getpass
import tempfile
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

# Construct a location in /tmp dir to hold cached data
dataPath = os.path.join(tempfile.gettempdir(), str(getpass.getuser()))
print(dataPath)
if not os.path.exists(dataPath):
    os.mkdir(dataPath)
filenameWithPath = os.path.join(dataPath, "mnist")

# Get training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data(path=filenameWithPath)

batch_size = 128
num_classes = 10
epochs = 20
# x_train: (60000, 28, 28)
# y_train: (60000,) = (60000)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# Each pixel: 0 < x < 1
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices (i.e. one-hots)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# y_train: (60000, 10)
# y_test: (10000, 10)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
# First dim of each (input/output) tensor represents bsize, so is left as None
breakpoint()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
# compile does nothing more than tell the model loss, optim and metrics

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# TODO print weights and biases of model
