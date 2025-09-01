from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# 🔹 Clases de los modelos
class_names_age = ["0_9","10_19","20_29","30_39","40_49","50_59","60_69","70_79","80_plus"]
class_names_gender = ["Mujer", "Hombre"]  

# 🔹 Cargar modelos
model_age = tf.keras.models.load_model(
    "ml/best_head.keras",
    custom_objects={"preprocess_input": preprocess_input}
)
model_gender = tf.keras.models.load_model(
    "ml/genero_head.keras",
    custom_objects={"preprocess_input": preprocess_input}
)

@app.route('/', methods=["GET"])
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            return "Archivo vacío o inválido", 400

        file_bytes = np.frombuffer(file.read(), np.uint8)
        if file_bytes.size == 0:
            return "Archivo vacío o inválido", 400

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return "Archivo inválido", 400

        # 🔹 Redimensionar y preprocesar para ambos modelos
        image_resized = cv2.resize(image, (380, 380))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_pre = preprocess_input(image_rgb.astype("float32"))
        image_input = np.expand_dims(image_pre, axis=0)

        # 🔹 Predicción de edad
        pred_age = model_age.predict(image_input)
        idx_age = int(np.argmax(pred_age, axis=1)[0])
        age_class = class_names_age[idx_age]
        age_conf = float(np.max(pred_age))

        # 🔹 Predicción de género
        pred_gender = model_gender.predict(image_input)
        idx_gender = int(np.argmax(pred_gender, axis=1)[0])
        gender_class = class_names_gender[idx_gender]
        gender_conf = float(np.max(pred_gender))

        # 🔹 Resultado combinado
        result = f"Edad Predicha: {age_class} (confianza: {age_conf:.2f}) | Género: {gender_class} (confianza: {gender_conf:.2f})"
        return result

    except Exception as e:
        # Captura cualquier error y lo muestra
        return f"Error interno: {str(e)}", 500


if __name__ == '__main__':
    import os
    if not os.environ.get("PYTEST_RUNNING"):  # evita que se ejecute en modo test
        app.run(debug=True, port=5002)

