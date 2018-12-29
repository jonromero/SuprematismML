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

SHAPE_SIZE=8
NUM_OF_SAMPLES=5000
RECTANGLE = 0
CIRCLE = 1

def image_from_data(image_data, test_coordinates, predict_coordinates, id='', show=False):
    test_coordinates = test_coordinates
    predict_coordinates = predict_coordinates

    tx, ty, tx1, ty1 = test_coordinates
    px, py, px1, py1 = predict_coordinates

    img = Image.fromarray(image_data.reshape(SHAPE_SIZE,SHAPE_SIZE), '1')
    img = img.convert("RGB")
    shape = ImageDraw.Draw(img)
    shape.rectangle(((tx,ty),(tx1,ty1)), outline="red")
    shape.rectangle(((px,py),(px1,py1)), outline="green")

    img.save('Data2Img/image_from_data-'+id+'-.png')
    if show:
        img.show()

def create_random_shape(id=None, save=False):    
    x1, y1 = np.random.randint(1, SHAPE_SIZE, size=2)
    x2 = np.random.randint(x1, SHAPE_SIZE)
    y2 = np.random.randint(y1, SHAPE_SIZE)
    
    canvas = Image.new("L", (SHAPE_SIZE, SHAPE_SIZE), "white")
    shape = ImageDraw.Draw(canvas)
    #type_of_shape = RECTANGLE # TODO: this need one-hot vector
    shape.rectangle(((x1,y1),(x2,y2)), fill="black")
    if save:
        canvas.save("Data/output"+str(id)+".png", "PNG")

    shape_np = np.array(canvas, dtype=np.float64)
    shape_np[shape_np == 0] = 1
    shape_np[shape_np == 255] = 0
    
    # return the image in numpy, the bounding box and the type
    return(shape_np, [x1, y1, x2, y2])

generated_data = map(create_random_shape, range(NUM_OF_SAMPLES))

# train data X is the image
# train data y is the validate data: shape and coordinates
data_input = np.array(map(lambda x: x[0], generated_data))
data_validate = np.array(map(lambda x: x[1:][0], generated_data))

# Normalize data
st_dev_data_input = np.std(data_input)
data_input = data_input.reshape(NUM_OF_SAMPLES, -1) / st_dev_data_input 
data_validate = data_validate.reshape(NUM_OF_SAMPLES, -1) / float(SHAPE_SIZE)

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
        Dense(400, activation='relu', input_dim=data_input.shape[-1]), 
        Dense(400),
        Dropout(0.2), 
        Dense(data_validate.shape[-1])
    ])

model.compile(optimizer='adadelta', loss='mean_squared_error')

# Train
model.fit(train_X, train_y, epochs=50, validation_data=(test_X, test_y), verbose=2)
model.summary()

# save the model
print("----------Saving the model------------")
model.save('shape_reko_'+str(SHAPE_SIZE)+'x'+str(SHAPE_SIZE)+'.h5')

# let's try out the model 
test_y_predictions = model.predict(test_X)

test_X = test_X * st_dev_data_input 
test_y = test_y * SHAPE_SIZE
test_y_predictions = np.round(test_y_predictions*SHAPE_SIZE)

print("first element of prediction")
print(test_y_predictions[0])
print(test_y[0])
print(test_X[0])
image_from_data(test_X[0], test_y[0], test_y_predictions[0], show=False)

# save the images of the predictions
for i in range(1, len(test_y_predictions)):
    image_from_data(test_X[i], test_y[i], test_y_predictions[i], id=str(i))

success_rate = (100 - (np.mean(test_y_predictions != test_y)*100))
print("Success rate (%): ", success_rate)