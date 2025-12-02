"""
Rutas de la aplicación web
"""

from flask import Blueprint, render_template, request, jsonify, current_app
import os
import sys
import traceback
from werkzeug.utils import secure_filename
from model.multi_detector import get_detector, ALGORITHM_CONFIG

main = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@main.route('/')
def index():
    """Página principal"""
    return render_template('index.html', algorithms=ALGORITHM_CONFIG)


@main.route('/test')
def test():
    """Página de prueba simple"""
    return render_template('test.html')


@main.route('/algorithms', methods=['GET'])
def get_algorithms():
    """Retorna lista de algoritmos disponibles"""
    return jsonify(ALGORITHM_CONFIG), 200


@main.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint para analizar una imagen subida"""
    try:
        # Verificar que se envió un archivo
        if 'image' not in request.files:
            return jsonify({'error': 'No se encontró ningún archivo'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Formato de archivo no permitido'}), 400
        
        # Obtener algoritmo seleccionado (default: ela)
        algorithm = request.form.get('algorithm', 'ela')
        
        # Guardar archivo de forma segura
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        print(f"[DEBUG] Guardando archivo en: {filepath}", file=sys.stderr)
        file.save(filepath)
        print(f"[DEBUG] Archivo guardado. Algoritmo: {algorithm}", file=sys.stderr)
        
        # Analizar imagen con el algoritmo seleccionado
        print(f"[DEBUG] Iniciando análisis con algoritmo: {algorithm}", file=sys.stderr)
        detector = get_detector()
        results = detector.analyze_image(filepath, algorithm=algorithm)
        print(f"[DEBUG] Análisis completado", file=sys.stderr)
        
        if 'error' in results:
            print(f"[ERROR] Error en análisis: {results['error']}", file=sys.stderr)
            return jsonify(results), 500
        
        return jsonify(results), 200
    
    except Exception as e:
        error_msg = f'Error al procesar la imagen: {str(e)}'
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        print(f"[ERROR] Traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return jsonify({'error': error_msg, 'details': traceback.format_exc()}), 500
