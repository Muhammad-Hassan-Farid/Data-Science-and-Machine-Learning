from flask import Flask, redirect, url_for, render_template, request
import cv2
import numpy as np
import tensorflow 
from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def result():
    res = ''
    img_size = (300, 300)
    lenet_model = load_model('lenet_model.h5')
    
    if request.method == 'POST':
        file = request.files['filename']
        if file:
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            
            image = np.array(img, dtype='float32')
            image = image / 255.0
            
            prediction = lenet_model.predict(np.expand_dims(image, axis=0))
            predicted_class = np.argmax(prediction)
            
            res = predicted_class
            if predicted_class == 1:
                res = 'Meningioma Tumor'
            elif predicted_class == 2:
                res = 'No Tumor'
            elif predicted_class == 0:
                res = 'Glioma tumor'
            elif predicted_class == 3:
                res = 'pituitary Tumor'
                
            
        else:
            res = "No file uploaded!"
            
        
        
        return render_template('index.html', result=res)

if __name__ == '__main__':
    app.run(debug=True)
