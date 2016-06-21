from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from skimage import io
import cv2
import os
import pandas as pd

def pop_layer(model):
    '''
    INPUTS: Keras Sequential Model Object
    OUTPUTS: None (Side Effect of Removing a Keras Layer)

    Takes a model as an input. If model doesn't have enough layers,
    raises exception. Clears layers after popping layer off.
    '''
    
    if not model.outputs:
        raise Exception("Model does not have enough layers to pop")

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes=[]
        model.outbound_nodes=[]
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

def VGG_16(weights):
    '''
    INPUT: H5 file of weights
    OUTPUT: Model fuction to be compiled

    Builds the VGG16 Model. Loads weights into the function. Drops the
    classification layer. Model will output a 4096 vector for each
    image
    '''
    model = Sequential()

    # Layer 1
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    # Layer 2
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    # Layer 3
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    # Layer 4
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    # Layer 5
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    # Flatten to Categorize
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Load Weights and Drop Classification Layer
    model.load_weights(weights)
   
    #Pop off Classification Layer and Final Dropout
    pop_layer(model) 
    pop_layer(model)
    
    return model

def transform_image(path):
    '''
    INPUTS: Path to Image (String)
    OUTPUTS: 4D Numpy Tensor

    Opens file with cv2, transposes to RGB, adds dimension and returns
    '''
    im = cv2.imread(path)
    im = im.transpose((2,1,0))
    return np.expand_dims(im, axis=0)

def create_weights(model):
    ''' 
    INPUTS: Model (Compiled Keras NN Function)
    OUTPUTS: Side Effects (Writes Pandas Dataframe to disk)

    Takes a compiled Keras NN function and feeds each image in
    given directory to the neural network. Output from each
    image is a (1,4096) numpy array, each of which is the weight
    on a neuron on the final connected layer. Writes these to file.
    '''
    data = pd.DataFrame()
    for x, image in enumerate(os.listdir('images/')):
        print "processing image {}".format(x)
        if image.endswith('.jpg'):
            img_arr = transform_image(os.getcwd() + '/images/' + image)
            output = model.predict(img_arr)[0]
            output.shape = (1,4096)
            data = pd.concat([data, pd.DataFrame(output)], axis=0)

    return data
if __name__ == "__main__":

    model = VGG_16('../transfer_learning/data/vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    data = create_weights(model)
    data.to_csv("cluster.csv", index=False)
