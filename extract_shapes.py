"""
# Tries to extract shapes from an image

Jon V
269th of December 2019
"""

import random

from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model

MAX_SHAPE_SIZE = 8

filename = "shape_test.png"

# convert image file to 1-d numpy array
# good luck with colors
# and make it smaller (the network expects a max input size)
img = Image.open(filename).convert('L')
img_np = np.array(img).reshape(MAX_SHAPE_SIZE, MAX_SHAPE_SIZE)
img_np = img_np  / np.std(img_np)
img_np = img_np.ravel()
img_np[img_np == 0] = 1
img_np[img_np > 2] = 0

img_np = np.array([img_np])

print img_np

# load the model for shape identification 
model = load_model('models/shape_reko_'+str(MAX_SHAPE_SIZE)+'x'+str(MAX_SHAPE_SIZE)+'.h5')

# let's look for coordinates
coordinates = model.predict(img_np)
coordinates = np.round(coordinates * MAX_SHAPE_SIZE)
print coordinates

def build_image(coordinates):    
    x1, y1, x2, y2 = coordinates

    canvas = Image.new("RGB", (MAX_SHAPE_SIZE, MAX_SHAPE_SIZE), "white")
    shape = ImageDraw.Draw(canvas)

    shape.rectangle(((x1,y1),(x2,y2)), fill="black")
    canvas.save("extracted_shape.png", "PNG")

# let's recreate the image
build_image(coordinates[0])

    
