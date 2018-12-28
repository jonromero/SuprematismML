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
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

SHAPE_SIZE=64
NUM_OF_SAMPLES=10000
RECTANGLE = 0
CIRCLE = 1

def show_image(image_data, test_coordinates, predict_coordinates):
    tx1, ty1, tx2, ty2, color = test_coordinates
    px1, py1, px2, py2, color = predict_coordinates

    img = Image.fromarray(image_data.reshape(SHAPE_SIZE,SHAPE_SIZE,4))

    shape = ImageDraw.Draw(img)
    shape.rectangle(((tx1,ty1),(tx2,ty2)), outline="red")
    shape.rectangle(((px1,py1),(px2,py2)), outline="green")

    img.save('my.png')
    img.show()

def create_random_shape(id=None, save=False):
    x1, y1, x2, y2 = random.sample(range(0, SHAPE_SIZE), 4)

    canvas = Image.new('RGBA', (SHAPE_SIZE, SHAPE_SIZE), "white")
    shape = ImageDraw.Draw(canvas)
    type_of_shape = RECTANGLE # TODO: this need one-hot vector
    shape.rectangle(((x1,y1),(x2,y2)), fill="black")
    if save:
        canvas.save("Data/output"+str(id)+".png", "PNG")

    shape_np = np.array(canvas)
    shape_np = shape_np.reshape(SHAPE_SIZE*SHAPE_SIZE*4)
    
    # return the image in numpy, the bounding box and the type
    return(shape_np, np.array([x1, y1, x2, y2, type_of_shape]))

generated_data = map(create_random_shape, range(NUM_OF_SAMPLES))

# train data X is the image
# train data y is the validate data: shape and coordinates
data_input = np.array(map(lambda x: x[0], generated_data))
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
        Dense(200, activation='relu', input_dim=data_input.shape[-1]), 
        Dense(200, activation='relu'),
        Flatten(),
        Dropout(0.2), 
        Dense(data_validate.shape[-1])
    ])
    
model.compile(optimizer='adadelta', loss='mse')

# Train
model.fit(train_X, train_y, epochs=30, validation_data=(test_X, test_y), verbose=2)
model.summary()

# save the model
#print("----------Saving the model------------")
#model.save('shape_reko.h5')

# let's try out the model 
test_y_predictions = model.predict(test_X)
print("first element of prediction")
print(test_y_predictions[0])
print(test_y[0])
print(test_X[0])
show_image(test_X[0], test_y[0], test_y_predictions[0])
