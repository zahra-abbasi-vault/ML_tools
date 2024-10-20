
import cv2
import tensorflow
import numpy as np

input_shape = [100,100,3]
image_channels = input_shape[-1]

def preprocess_image(image):
    """
    this is done after augmentation, just before feeding to network"""
    image = cv2.resize(image, (input_shape[1], input_shape[0]))
    if image_channels==1 and len(image.shape)==3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
    if image_channels==1: image = np.expand_dims(image, axis=2)
    if image_channels == 3 : image = image / 255
    return image


image_path = r"path/to/img"
image = cv2.imread(image_path)
pp_image = preprocess_image(image)
pp_image = np.expand_dims(pp_image, axis=0)
model_path = r"path/to/tensorflow/model" 
model = tensorflow.keras.models.load_model(model_path)
model.summary()
x = model.predict(pp_image)
print(x)
...

