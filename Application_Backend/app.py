from flask import Flask, request, jsonify
import base64
from PIL import Image
from io import BytesIO
import io
import uuid
import os


app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    image_base64 = data.get('image')

    if not image_base64:
        return {"error": "No image data"}

    # Decode the base64 string
    image_data = base64.b64decode(image_base64)

    # Save the image
    with open(os.path.join('images', 'image.jpg'), 'wb') as f:
        f.write(image_data)

    return {"class":"tomato","confidence": "0.06"}


# flask run -h 192.168.8.190 -p 3000 --debug