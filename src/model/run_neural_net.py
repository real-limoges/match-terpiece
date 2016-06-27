import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32,cnmem=1"
import theano

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from skimage import io
import pandas as pd
from scipy import misc

DATA_DIR = 'data/'
IMAGE_DIR = 'images/'

def pop_layer(model):
    '''
    INPUTS: Keras Sequential Model Object
    OUTPUTS: None (Side Effect of Removing a Keras Layer)

    Takes a model as an input. If model doesn't have enough layers,
    raises exception. Clears layers after popping layer off.
    '''

    if not model.outputs:
        raise Exception('Model does not have enough layers to pop')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False


def VGG_16(weights, cls='fc1'):
    '''
    INPUT: H5 file of weights
    OUTPUT: Model fuction to be compiled

    Builds the VGG16 Model. Loads weights into the function. Drops the
    classification layer. Model will output a 4096 vector for each
    image
    '''
    model = Sequential()

    # Layer 1
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Layer 2
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Layer 3
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Layer 4
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Layer 5
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Flatten to Categorize
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Load Weights and Drop Classification Layer
    model.load_weights(weights)

    if cls == 'fc1':
        for _ in xrange(4):
            pop_layer(model)
    elif cls == 'fc2':
        for _ in xrange(2):
            pop_layer(model)

    return model


def transform_image(path):
    '''
    INPUTS: Path to Image (String)
    OUTPUTS: 4D Numpy Tensor

    Opens file with cv2, transposes to RGB, adds dimension and returns
    '''
    mean_pixel = [103.939, 116.779, 123.68]

    if os.stat(path).st_size < 4000:
        return None

    try:

        im = misc.imread(path)
        im = im[:,:,[2,1,0]]
        im = misc.imresize(im, (224, 224)).astype(np.float32)

        for channel in xrange(3):
            im[:,:,channel] -= mean_pixel[channel]
        im = im.transpose((2, 1, 0))
    
        return np.expand_dims(im, axis=0)
    except Exception:
        print "Issues with file {}".format(path)
        return None


def create_weights(model, cls):
    '''
    INPUTS: Model (Compiled Keras NN Function)
    OUTPUTS: Side Effects (Writes Pandas Dataframe to disk)

    Takes a compiled Keras NN function and feeds each image in
    given directory to the neural network. Output from each
    image is a (1,4096) numpy array, each of which is the weight
    on a neuron on the final connected layer. Writes these to file.
    '''
    data = pd.DataFrame()
    image_names = []

    image_list = [image for image in os.listdir(IMAGE_DIR)
                        if image.endswith('.jpg')]
    for x, image in enumerate(image_list):
        if image.endswith('.jpg'):
            img_arr = transform_image(IMAGE_DIR + image)
            if img_arr is not None:
                image_names.append(image)
                print 'processing image {} for {}'.format(x, cls)
                output = model.predict(img_arr)[0]
                output.shape = (1, 4096)
                data = pd.concat([data, pd.DataFrame(output)], axis=0)
    data.set_index(pd.Series(image_names), inplace=True)
    
    return data


if __name__ == '__main__':

    model_fc1 = VGG_16(DATA_DIR + 'vgg16_weights.h5', 'fc1')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model_fc1.compile(optimizer=sgd, loss='categorical_crossentropy')

    model_fc2 = VGG_16(DATA_DIR + 'vgg16_weights.h5', 'fc2')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model_fc2.compile(optimizer=sgd, loss='categorical_crossentropy')

    data_fc1 = create_weights(model_fc1, 'fc1')
    data_fc1.to_csv(DATA_DIR + 'fc1_cluster.csv')
    
    data_fc2 = create_weights(model_fc2, 'fc2')
    data_fc2.to_csv(DATA_DIR + 'fc2_cluster.csv')
