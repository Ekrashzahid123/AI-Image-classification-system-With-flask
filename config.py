# config.py
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

SECRET_KEY = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'users.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False
