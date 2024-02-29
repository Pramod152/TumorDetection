from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
import base64

app = Flask(__name__)

# Define a global variable to store the loaded model
loaded_model = None

# Define a function to load the model
def load_model():
    global loaded_model
    if loaded_model is None:
        loaded_model = joblib.load('svm_model.pkl')

# Define image processing and prediction function
def process_image(image):
    load_model()  # Ensure the model is loaded
    # Your image processing and prediction logic here

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            try:
                # Read image
                image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                # Process image and make prediction
                prediction = process_image(image)
                # Encode image to base64
                _, img_encoded = cv2.imencode('.jpg', image)
                img_src = 'data:image/jpg;base64,' + base64.b64encode(img_encoded).decode()
                # Render result template with prediction and image source
                return render_template('result.html', result=prediction, img_src=img_src)
            except Exception as e:
                # Log any exceptions that occur during image processing
                print(f"Error processing image: {str(e)}")
                return render_template('result.html', result="Error processing image", img_src="")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
