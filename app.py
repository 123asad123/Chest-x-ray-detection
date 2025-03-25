from flask import Flask, render_template, request, flash, redirect, url_for
import urllib.request
import os
from werkzeug.utils import secure_filename
# from keras.preprocessing.image import load_imag
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
# from keras.applications.vgg16 import VGG16

from tensorflow.keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')



@app.route('/', methods=['POST'])
def predict():
    # imagefile = request.files['imagefile']
    # image_path = "./images/" + imagefile.filename
    # print(image_path)
    # imagefile.save(image_path)

    file = request.files['imagefile']
    # if 'file' not in request.files:
    #     flash('No file part')
    #     return redirect(request.url)
    # file = request.files['file']
    # if file.filename == '':
    #     flash('No image selected for uploading')
    #     return redirect(request.url)

    #image_path = "./static/uploads" + file.filename

    model=load_model("E:/FYP/FYP_2024/Deployment/Copy of model_vgg16.h5")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
    #model = VGG16(weights='image
    # net', include_top=False)

    
    #img_path = '/content/drive/MyDrive/Chest X ray dataset/val/PNEUMONIA/person1951_bacteria_4882.jpeg'
    image_path = "./static/uploads/" + file.filename
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)

    listToStr = ' '.join(map(str, features))
    x = listToStr.split()
    normal = x[0].replace("[", "")
    pneumonia = x[1].replace("]", "")
    if normal>pneumonia:
        c="normal"
    else:
        c="pneumonia"

    Disclaimer = "not empty"
    return render_template('index.html',prediction=c,textone="Our predicted Image is",filename=filename,texttwo="Conclusion : ",Disclaimer=Disclaimer)


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 


if __name__== '__main__':
    app.run(port=3000, debug=True)
