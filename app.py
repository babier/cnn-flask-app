from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model('model/model.h5')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def prepare_image(image_path):
    img = Image.open(image_path).resize((32,32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = prepare_image(filepath)
            pred = model.predict(img)
            prediction = class_names[np.argmax(pred)]

            return render_template('index.html', prediction=prediction, image=filepath)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)