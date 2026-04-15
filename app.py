from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import os
import cv2
import requests
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import pandas as pd

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

# ---------------------------------
# CONFIG
# ---------------------------------

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URL", "sqlite:///app.db")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ---------------------------------
# MODELOS (DINÁMICOS)
# ---------------------------------

from ultralytics import YOLO

MODEL_URL = "https://drive.google.com/file/d/1qUGwEr5XLgaAx4toXkJUzM5X2hG2CWlb/view?usp=sharing"
CLASSIFIER_URL = "https://drive.google.com/file/d/1DpYI0MmBW8mGKoArE50byUbpYB7NJUck/view?usp=sharing"

MODEL_PATH = "model/best.pt"
CLASSIFIER_PATH = "model/frotis_classifier.pt"

model = None
classifier = None


def download_models():
    os.makedirs("model", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo detección...")
        r = requests.get(MODEL_URL)
        open(MODEL_PATH, "wb").write(r.content)

    if not os.path.exists(CLASSIFIER_PATH):
        print("Descargando clasificador...")
        r = requests.get(CLASSIFIER_URL)
        open(CLASSIFIER_PATH, "wb").write(r.content)


def load_models():
    global model, classifier

    if model is None or classifier is None:
        download_models()

        if model is None:
            print("Cargando modelo detección...")
            model = YOLO(MODEL_PATH)
            model.to("cpu")

        if classifier is None:
            print("Cargando clasificador...")
            classifier = YOLO(CLASSIFIER_PATH)
            classifier.to("cpu")


# ---------------------------------
# FUNCIONES
# ---------------------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_blood_smear(image_path):
    load_models()

    results = classifier(image_path)
    predicted = results[0].names[results[0].probs.top1]
    confidence = float(results[0].probs.top1conf)

    return predicted.lower() == "frotis" and confidence > 0.75


def estimate_platelets_per_ul(count):
    return count * 15000


def classify_dengue_risk(platelets):
    if platelets >= 150000:
        return "Normal"
    elif platelets >= 100000:
        return "Riesgo leve"
    elif platelets >= 50000:
        return "Posible dengue"
    else:
        return "Dengue severo"


# ---------------------------------
# MODELOS DB
# ---------------------------------

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    role = db.Column(db.String(50), default="Usuario")


class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dni = db.Column(db.String(8), unique=True)
    nombres = db.Column(db.String(150))
    apellidos = db.Column(db.String(150))


class PlateletResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(150))
    platelet_count = db.Column(db.Integer)
    platelets_estimated = db.Column(db.Integer)
    dengue_status = db.Column(db.String(50))
    analysis_date = db.Column(db.DateTime, default=lambda: datetime.utcnow() - timedelta(hours=5))

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'))

    patient = db.relationship('Patient')


# ---------------------------------
# INIT
# ---------------------------------

with app.app_context():
    db.create_all()

# ---------------------------------
# LOGIN
# ---------------------------------

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()

        if user and bcrypt.check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('index'))

        flash("Error login", "danger")

    return render_template("login.html")


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ---------------------------------
# PACIENTES API
# ---------------------------------

@app.route('/api/patient', methods=['POST'])
@login_required
def create_patient():
    data = request.json

    dni = data.get('dni')
    nombres = data.get('nombres')
    apellidos = data.get('apellidos')

    if not dni or not dni.isdigit() or len(dni) != 8:
        return jsonify({"error": "DNI inválido"}), 400

    if Patient.query.filter_by(dni=dni).first():
        return jsonify({"error": "Ya existe"}), 400

    patient = Patient(dni=dni, nombres=nombres, apellidos=apellidos)
    db.session.add(patient)
    db.session.commit()

    return jsonify({"id": patient.id})


@app.route('/api/patient/<dni>')
@login_required
def get_patient(dni):
    patient = Patient.query.filter_by(dni=dni).first()

    if not patient:
        return jsonify({"error": "No encontrado"}), 404

    return jsonify({
        "id": patient.id,
        "nombres": patient.nombres,
        "apellidos": patient.apellidos
    })


# ---------------------------------
# DETECCIÓN
# ---------------------------------

@app.route('/count-platelets', methods=['POST'])
@login_required
def count_platelets():

    load_models()

    file = request.files['image']
    patient_id = request.form.get('patient_id')

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(path)

    if not is_blood_smear(path):
        return jsonify({"error": "No es frotis"}), 400

    image = cv2.imread(path)
    results = model(image)

    count = 0

    for box in results[0].boxes:
        if int(box.cls[0]) == 1:
            count += 1

    est = estimate_platelets_per_ul(count)
    dengue = classify_dengue_risk(est)

    db.session.add(PlateletResult(
        image_filename=filename,
        platelet_count=count,
        platelets_estimated=est,
        dengue_status=dengue,
        user_id=current_user.id,
        patient_id=patient_id
    ))

    db.session.commit()

    return jsonify({
        "platelets_detected": count,
        "estimated_platelets_per_ul": est,
        "dengue_status": dengue
    })


# ---------------------------------
# MAIN
# ---------------------------------

@app.route('/')
@login_required
def index():
    return render_template("index.html")


# ---------------------------------
# RUN
# ---------------------------------

if __name__ == '__main__':
    app.run()
