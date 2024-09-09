import os
from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import base64
import uuid

app = Flask(__name__)

# Load the pre-trained model
model = load_model('D:/age_gender/model.h5')

# Gender labels
gender_dict = {0: 'Male', 1: 'Female'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    uploaded_file = request.files['image']
    
    # Generate a unique filename
    unique_filename = str(uuid.uuid4())
    image_path = f'temp_{unique_filename}.jpg'
    
    # Save the uploaded image to a temporary location
    uploaded_file.save(image_path)
    
    # Read the image file and convert it to grayscale
    img = Image.open(image_path).convert('L')
    
    # Resize the image
    img = img.resize((128, 128))
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    # Reshape and normalize the image for model input
    img_array = img_array.reshape(1, 128, 128, 1) / 255.0
    
    # Predict gender and age
    pred_gender, pred_age = model.predict(img_array)
    pred_gender = gender_dict[int(np.round(pred_gender))]
    pred_age = int(np.round(pred_age))
    
    # Convert the image data to base64 string
    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Remove the temporary image file
    os.remove(image_path)
    
    return render_template('result.html', gender=pred_gender, age=pred_age, image_data=image_data)

if __name__ == '__main__':
    app.run(debug=True)
