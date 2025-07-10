from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions, MobileNetV2
from tensorflow.keras.preprocessing import image
import urllib.request
import os

app = Flask(__name__)
IMG_SIZE = 224
model = MobileNetV2(weights='imagenet')

sample_images = {
    "Basmati": "https://www.amoliinternational.com/images/basmati/1121-row.jpeg",
    "Brown": "https://tse4.mm.bing.net/th/id/OIP.IkeJEKlkqujCkmuZ4XoU4gHaE6?pid=Api&P=0&h=180",
    "Arborio": "https://thumbs.dreamstime.com/b/white-italian-arborio-rice-used-making-risotto-dish-close-up-261740175.jpg"
}

@app.route('/')
def index():
    return render_template('index.html', sample_images=sample_images)

@app.route('/predict', methods=['POST'])
def predict():
    img_url = request.form['img_url']
    img_path = "static/temp.jpg"
    urllib.request.urlretrieve(img_url, img_path)

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]

    return render_template('result.html', predictions=decoded, image_url=img_url)

if __name__ == '__main__':
    app.run(debug=True)
