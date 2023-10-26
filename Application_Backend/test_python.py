from flask import Flask, request, jsonify
import base64
from PIL import Image
from io import BytesIO
import os
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


IMAGE_SIZE=256
model = tf.keras.models.load_model("./models/tomato_v2.h5")

class_names=['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

def predict_from_image_path(image_path):
    
    # Load the image
    img = image.load_img(image_path, target_size=(256, 256))

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand dimensions to fit the model's input shape
    img_array = np.expand_dims(img_array, axis=0)

    # Use the model to make a prediction
    prediction = model.predict(img_array)

    # Get the predicted class and confidence
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 * (np.max(prediction[0])), 2)
    
    return predicted_class, confidence

image_path = "TestImages/0a7cc59f-b2b0-4201-9c4a-d91eca5c03a3___PSU_CG 2230.JPG"

predicted_class, confidence = predict_from_image_path(image_path)

print(f"class:{predicted_class},confidence: {confidence}")