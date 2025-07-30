from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2

# Creating a Flask Instance
app = Flask(__name__)

IMAGE_SIZE = (150, 150)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading Pre-trained Model ...")
model = load_model('model.h5')


def image_preprocessor(path):
    currImg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
    currImg = cv2.resize(currImg, IMAGE_SIZE)
    currImg = currImg / 255.0  # Normalize
    return np.expand_dims(currImg, axis=(0, -1))  # Shape: (1, 150, 150, 1)



def model_pred(image):
    '''
    Perform prediction on the preprocessed image.
    '''
    print("Image shape:", image.shape)
    prediction = model.predict(image)[0][0]
    return int(prediction > 0.5)  # Return 1 if Pneumonia, 0 if Normal


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'imageFile' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['imageFile']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            imgPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(imgPath)
            print(f"Image saved at {imgPath}")

            image = image_preprocessor(imgPath)
            pred = model_pred(image)

            return render_template('upload.html', name=filename, result=pred)
        else:
            flash('Invalid file format. Please upload JPG, JPEG, or PNG.')
            return redirect(request.url)

    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
