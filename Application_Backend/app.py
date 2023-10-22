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


app = Flask(__name__)

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
    
    #image_path = "./Dataset/tomato/Tomato__Target_Spot/0a2de4c5-d688-4f9d-9107-ace1d281c307___Com.G_TgS_FL 7941.JPG"

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


@app.route("/predict", methods=['POST'])
def predict():
    print("called")
    data = request.get_json()
    image_base64 = data.get('image')

    if not image_base64:
        return {"error": "No image data"}

    # Decode the base64 string
    image_data = base64.b64decode(image_base64)

    # Save the image
    with open(os.path.join('images', 'image.jpg'), 'wb') as f:
        f.write(image_data)

    image_path = "./images/image.jpg"

    predicted_class, confidence = predict_from_image_path(image_path)

    return {"class":predicted_class,"confidence": confidence}


# myenv/Scripts/activate

# flask run -h 192.168.8.190 -p 3000 --debug
