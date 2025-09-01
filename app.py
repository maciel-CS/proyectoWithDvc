from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# Clases reales (ajusta con tu dataset)
class_names = ["0_9","10_19","20_29","30_39","40_49","50_59","60_69","70_79","80_plus"]

# Cargar modelo
model = tf.keras.models.load_model(
    "ml/best_head.keras",
    custom_objects={"preprocess_input": preprocess_input}
)

@app.route('/', methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file or file.filename == "":
        return "Archivo vacío o inválido", 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    
    # 🔹 Validar antes de usar OpenCV
    if file_bytes.size == 0:
        return "Archivo vacío o inválido", 400

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return "Archivo vacío o inválido", 400

    # Procesamiento normal
    image = cv2.resize(image, (380, 380))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image.astype("float32"))
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    pred_idx = int(np.argmax(prediction, axis=1)[0])
    pred_cls = class_names[pred_idx]
    conf = float(np.max(prediction))

    return f"Edad de Predicción: {pred_cls} (confianza: {conf:.2f})"


if __name__ == '__main__':
    import os
    if not os.environ.get("PYTEST_RUNNING"):  # evita que se ejecute en modo test
        app.run(debug=True, port=5002)

