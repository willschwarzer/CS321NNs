import os
import sys
import os.path
import getpass
import tempfile
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


if sys.argv[2] == '0':
    
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

else:

    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    batch_size = 128
    num_classes = 10
    epochs = 20
    axis = list(x_train.shape).index(3)
    x_train = x_train.sum(axis=axis)
    x_train = x_train.reshape(-1, 1024)
    axis = list(x_test.shape).index(3)
    x_test = x_test.sum(axis=axis)
    x_test = x_test.reshape(-1, 1024)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255*3
    x_test /= 255*3
    # Each pixel: 0 < x < 1
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices (i.e. one-hots)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)



# Neural net with no hidden layers
simple_net = Sequential()
simple_net.add(Dense(num_classes, activation='softmax',input_shape=(x_train.shape[1],)))

# Neural net with 1 hidden layer

single_layer_net = Sequential()
single_layer_net.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
single_layer_net.add(Dropout(0.2))
single_layer_net.add(Dense(num_classes, activation='softmax'))

# Neural net with 2 hidden layers

double_layer_net = Sequential()
double_layer_net.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
double_layer_net.add(Dropout(0.2))
double_layer_net.add(Dense(512, activation='relu'))
double_layer_net.add(Dropout(0.2))
double_layer_net.add(Dense(num_classes, activation='softmax'))


if sys.argv[1] == '0':
    model = simple_net
else:
    model = single_layer_net

model.summary()
# First dim of each (input/output) tensor represents bsize, so is left as None


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

if sys.argv[1] == '0':
    layer = model.layers[0]
    numpy_array = layer.get_weights()[0]
    numpy_array = numpy_array.transpose()
    #print(len(numpy_array))
    for i in range(10):
        if sys.argv[2] == '0':
            size = 28
        else:
            size = 32
        numpy_square = numpy_array[i].reshape(size,size)
        plt.imshow(numpy_square, cmap="Greys")
        plt.savefig('digit'+str(i)+'.png')
else:
    
    layer = model.layers[0]
    numpy_array = layer.get_weights()[0]
    #numpy_array = numpy_array.transpose()
    #1024 > 512
    #512 > 10
    hidden_layer = model.layers[1].get_weights()[0]
    first_layer = model.layers[0].get_weights()[0]
    
    array = first_layer @ hidden_layer



    for i in range(10):
        if sys.argv[2] == '0':
            size = 28
        else:
            size = 32
        numpy_square = array[i].reshape(size,size)
        plt.imshow(numpy_square, cmap="Greys")
        plt.savefig('digit'+str(i)+'.png')

    
# TODO print weights and biases of model
