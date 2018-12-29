"""
# Tries to extract shapes from an image

Jon V
269th of December 2019
"""

import random

from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model

filename = "shape_test.png"

# convert image file to 1-d numpy array
# good luck with colors
# and make it smaller (the network expects a max input size)

img = Image.open(filename).convert('RGBA')
img_np = np.array(img)
img_np = img_np.ravel()

print img_np

# load the model for shape identification 
model = load_model('shape_reko.h5')

# let's look for coordinates
coordinates = model.predict(img_np)
print coordinates

# let's recreate the image
build_image(coordinates)
