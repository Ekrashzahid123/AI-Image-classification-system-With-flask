from flask import Flask, render_template, request, jsonify
from flask_login import LoginManager, login_required
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

from config import SECRET_KEY, UPLOAD_FOLDER, SQLALCHEMY_DATABASE_URI, SQLALCHEMY_TRACK_MODIFICATIONS
from models import db, User
from auth import auth

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = SQLALCHEMY_TRACK_MODIFICATIONS

# --- Initialize DB ---
db.init_app(app)
with app.app_context():
    db.create_all()

# --- Ensure Upload Folder Exists ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Flask-Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Register Blueprint for Auth Routes ---
app.register_blueprint(auth)

# --- Routes ---
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    filename = secure_filename(image.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(save_path)

    # Preprocess
    img = Image.open(save_path).convert('RGB').resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    preds = VGG19(weights='imagenet').predict(img_array)
    decoded = decode_predictions(preds, top=3)[0]
    results = [{'label': label, 'probability': float(prob)} for (_, label, prob) in decoded]
    return jsonify(results)

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)
