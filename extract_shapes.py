"""
# Tries to extract shapes from an image

Jon V
269th of December 2019
"""

import random

from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

MAX_SHAPE_SIZE = 32

filename = "test_extract_data/shape_test.png"

#filename = "test_extract_data/test.png"

# convert image file to 1-d numpy array
# good luck with colors
# and make it smaller (the network expects a max input size)
img = Image.open(filename).convert('L')
old_size = img.size  

desired_size = MAX_SHAPE_SIZE
ratio = float(desired_size)/max(old_size)
new_size = tuple([int(x*ratio) for x in old_size])

img = img.resize(new_size)

delta_w = desired_size - new_size[0]
delta_h = desired_size - new_size[1]
padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
img = ImageOps.expand(img, padding)

img.save("test_extract_data/new.png")

img_np = np.array(img).reshape(MAX_SHAPE_SIZE, MAX_SHAPE_SIZE)
img_np[img_np == 0] = 1
img_np[img_np > 1] = 0
img_np = img_np.reshape(1, -1)

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
    canvas.save("test_extract_data/extracted_shape.png", "PNG")

# let's recreate the image
build_image(coordinates[0])

    
