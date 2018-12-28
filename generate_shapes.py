"""
# Generates a lot of images, based on random polygon 
# and saves the image+metadate to the filename.
# The idea is that by generating a ton of shapes, we can train a model
# to first find the boundind box, clasify them and then draw again these shapes

# Version 0.1, tries to recognize black rectangles and find their bounding box
# Version 0.2 will try with triangles
# Version 0.3 with colors
# Version 0.4 will try to recreate the original image

Jon V
26th of December 2019
"""

import random

from PIL import Image, ImageDraw
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

SHAPE_SIZE=8
NUM_OF_SAMPLES=50
RECTANGLE = 0
CIRCLE = 1

def create_random_shape(id=None):
    x1, y1, x2, y2 = random.sample(range(0, SHAPE_SIZE), 4)

    canvas = Image.new('RGBA', (SHAPE_SIZE, SHAPE_SIZE), "white")
    shape = ImageDraw.Draw(canvas)
    type_of_shape = RECTANGLE
    shape.rectangle(((x1,y1),(x2,y2)), fill="black")
    canvas.save("Data/output"+str(id)+".png", "PNG")

    shape_np = np.array(canvas)

    # return the image in numpy, the bounding box and the type
    return(shape_np, np.array([x1, y1, x2, y2, type_of_shape]))

generated_data = map(create_random_shape, range(NUM_OF_SAMPLES))

# train data X is the image
# train data y is the validate data: shape and coordinates
data_input = np.array(map(lambda x: x[1], generated_data))
data_validate = np.array(map(lambda x: x[1:][0], generated_data))

def create_train_set(data):
    # use 80% for train, 20% for test
    i = int(0.8 * NUM_OF_SAMPLES)
    train_set = data[:i]
    test_test = data[i:]
    
    return train_set, test_test

train_X, test_X = create_train_set(data_input)
train_y, test_y = create_train_set(data_validate)

# Build the model.
model = Sequential([
        Dense(200, input_dim=data_input.shape[-1]), 
        Activation('relu'), 
        Dropout(0.2), 
        Dense(data_validate.shape[-1])
    ])

model.compile(optimizer='adadelta', loss='mse')

# Train.
model.fit(train_X, train_y, epochs=30, validation_data=(test_X, test_y), verbose=2)

# save the model
print("----------Saving the model------------")
model.save('shape_reko.h5')
