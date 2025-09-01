import os
os.environ["PYTEST_RUNNING"] = "1" 

import io
import pytest
from app import app

# Cliente de prueba de Flask
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# ===== Prueba de humo =====
def test_api_disponible(client):
    """Verifica que la API está disponible en '/'"""
    response = client.get('/')
    assert response.status_code == 200
    # Decodificar bytes a string UTF-8 antes de comparar
    response_text = response.data.decode("utf-8")
    assert "Predicciones de Edad y Género" in response_text


# ===== Prueba de un golpe =====
def test_prediccion_uno(client):
    """Enviar una sola imagen válida y verificar predicción"""
    with open("tests/person2.png", "rb") as f:  # colocá una imagen conocida
        data = {"file": (io.BytesIO(f.read()), "sample_image.jpg")}
        response = client.post("/predict", content_type='multipart/form-data', data=data)
        assert response.status_code == 200
        # Verificar que la respuesta contiene una clase esperada
        assert any(cls.encode() in response.data for cls in [
            "0_9","10_19","20_29","30_39","40_49","50_59","60_69","70_79","80_plus"
        ])

# ===== Prueba de borde =====
@pytest.mark.parametrize("file_content", [
    b"",                      # archivo vacío
    b"not an image",          # contenido inválido
])
def test_prediccion_borde(client, file_content):
    """Enviar casos extremos y esperar error 400"""
    data = {"file": (io.BytesIO(file_content), "fake.jpg")}
    response = client.post("/predict", content_type='multipart/form-data', data=data)
    assert response.status_code == 400

# ===== Prueba de patrón =====
def test_patron_consistencia(client):
    """Verificar consistencia en varias imágenes similares"""
    predictions = []
    for i in range(3):  # repetimos varias veces la misma imagen
        with open("tests/person5.png", "rb") as f:
            data = {"file": (io.BytesIO(f.read()), f"sample_{i}.jpg")}
            response = client.post("/predict", content_type='multipart/form-data', data=data)
            assert response.status_code == 200
            predictions.append(response.data.decode())
    # Verificamos que las predicciones sean iguales
    assert len(set(predictions)) == 1
