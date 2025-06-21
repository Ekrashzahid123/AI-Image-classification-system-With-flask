from flask import Flask, request, jsonify, render_template
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- Config ---
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load Pretrained Model ---
model = VGG19(weights='imagenet')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    filename = secure_filename(image.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(save_path)

    # Preprocess image
    img = Image.open(save_path).convert('RGB').resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=3)[0]

    results = [{'label': label, 'probability': float(prob)} for (_, label, prob) in decoded]
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
