from flask import Flask, request, render_template
import tensorflow as tf
import os
import cv2
import numpy as np

app = Flask(__name__)

# Cargar el modelo TensorFlow al iniciar
model = tf.keras.models.load_model("ml/my_model.keras")


@app.route('/', methods=["GET", "POST"])
def home():
    return render_template("index.html")


@app.route('/predict', methods=["POST"])
def predict():
    # Validar si viene archivo
    if "file" not in request.files:
        return "No se encontró archivo en la petición", 400

    file = request.files["file"]

    if file.filename == "":
        return "Archivo vacío", 400

    # Leer bytes de la imagen
    file_bytes = np.frombuffer(file.read(), np.uint8)

    if file_bytes.size == 0:
        return "Archivo vacío o inválido", 400

    # Decodificar imagen
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Manejar errores de lectura
    try:
        image = preprocess_image(image)
    except Exception as e:
        return f"Archivo inválido. Error: {str(e)}", 400

    # Hacer predicción
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return f"Es un {predicted_class} ;) !!"

def preprocess_image(image):
    if image is None:
        raise ValueError("No se pudo leer la imagen. Puede estar vacía o tener formato inválido.")

    # Redimensionar a 28x28 (igual que MNIST)
    image = cv2.resize(image, (28, 28))
    # Normalizar a [0,1]
    image = image.astype("float32") / 255.0
    # Expandir dimensiones para que sea (1,28,28,1)
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image


if __name__ == '__main__':
    app.run(debug=True, port=5002)
