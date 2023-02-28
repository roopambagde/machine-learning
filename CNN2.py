import tensorflow as tf
from PIL import Image
import numpy as np

# Load the image and resize it
image = Image.open("Cat.jpg")
image = image.resize((32, 32))

# Convert the image to a numpy array
image_array = np.array(image)

# Reshape the array to add the batch dimension
image_array = image_array.reshape((1, 32, 32, 3))

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Make a prediction on the image
prediction = model.predict(image_array)

# Print the prediction
print(prediction)
