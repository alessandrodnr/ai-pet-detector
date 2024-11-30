from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
from werkzeug.utils import secure_filename, quote  # Actualiza la importación aquí
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuración de la carpeta de carga y extensiones permitidas
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Crear carpeta uploads si no existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Función para verificar si el archivo tiene una extensión permitida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Ruta para servir las imágenes subidas
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Cargar el modelo
try:
    model = load_model('modelo_perros_gatos_vgg16.h5')
    logger.info("Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")
    raise

# Preprocesar la imagen
def prepare_image(img_file):
    try:
        img = Image.open(img_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(-1, 150, 150, 3)
        return img_array
    except Exception as e:
        logger.error(f"Error en prepare_image: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Guardar la imagen en la carpeta uploads con un nombre seguro
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocesar la imagen
            img_array = prepare_image(file_path)

            # Realizar predicción
            prediction = model.predict(img_array)

            # Decidir el resultado
            result = "Perro" if prediction[0][0] > 0.5 else "Gato"
            confidence = prediction[0][0] if result == "Perro" else 1 - prediction[0][0]

            logger.info(f"Predicción final: {result} con confianza de {confidence:.2%}")

            # Enviar a la plantilla la imagen subida y el resultado
            return render_template('index.html', 
                                   prediction=f"{result} (Confianza: {confidence:.2%})", 
                                   filename=filename)

        except Exception as e:
            logger.error(f"Error en el proceso de predicción: {str(e)}")
            return jsonify({'error': f'Error procesando la imagen: {str(e)}'}), 500

    else:
        return jsonify({'error': 'Archivo no permitido. Solo imágenes (png, jpg, jpeg, gif) son aceptadas.'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
