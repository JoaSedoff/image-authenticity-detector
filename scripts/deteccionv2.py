import cv2
import numpy as np
import os
import glob
import json
import sys

# Agregar directorio raíz del proyecto al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuración de umbrales determinada por análisis de dataset_features.csv (Actualizado)
# Best Feature: laplacian_mean
# Best Threshold: 4.5853
# Best Accuracy: 68.52%
# Direction: higher_is_fake

CLASSIFICATION_CONFIG = {
    "primary_feature": "laplacian_score", # Cambiado dinámicamente
    "threshold": 4.5853,
    "direction": "higher_is_fake"
}

def classify_metrics(metrics):
    """
    Clasifica una imagen basada en sus métricas y la configuración actual.
    Retorna (predicción, confianza)
    """
    # Seleccionar la métrica correcta basada en la configuración
    # Mapeo de nombres de features del CSV a nombres en el diccionario 'metrics'
    feature_map = {
        "laplacian_mean": "laplacian_score",
        "fft_ratio_af_bf": "ratio_af_bf",
        "fft_uniformidad": "uniformidad_radial",
        "fft_entropia": "entropia",
        "fft_ratio_media": "ratio_freq_media"
    }
    
    # Determinar qué métrica usar (hardcoded por ahora basado en auto_calibrate)
    # En este caso, auto_calibrate eligió 'laplacian_mean'
    metric_key = "laplacian_score" 
    val = metrics.get(metric_key, 0)
    
    threshold = CLASSIFICATION_CONFIG["threshold"]
    direction = CLASSIFICATION_CONFIG["direction"]
    
    is_fake = False
    if direction == "higher_is_fake":
        if val > threshold:
            is_fake = True
    else: # lower_is_fake
        if val < threshold:
            is_fake = True
            
    if is_fake:
        prediction = "GENERADA POR IA"
        # Confianza simple
        dist = abs(val - threshold)
        # Normalizar confianza un poco (heurística)
        confidence = min(99.0, 60.0 + (dist * 10)) 
    else:
        prediction = "REAL"
        dist = abs(val - threshold)
        confidence = min(99.0, 60.0 + (dist * 10))
        
    return prediction, round(confidence, 2)

# Asegurarse de que podemos importar desde el directorio actual
sys.path.append(os.getcwd())

# Importación diferida para evitar ciclos si detector.py importa este módulo
# try:
#     from model.detector import AIImageDetector
# except ImportError:
#     ...

class AutoDetectorV2:
    def __init__(self):
        # Importación local para evitar ciclo de importación
        try:
            from model.detector import AIImageDetector
        except ImportError:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from model.detector import AIImageDetector
            
        self.detector = AIImageDetector()
        self.results = []
        self.results = []

    def analyze_directory(self, directory_path, label=None):
        """
        Analiza todas las imágenes en un directorio de forma recursiva.
        
        Args:
            directory_path: Ruta al directorio de imágenes
            label: Etiqueta opcional para las imágenes ('real', 'fake', etc.)
        """
        print(f"Analizando directorio: {directory_path} (Etiqueta: {label})")
        valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
        files = []
        
        # Búsqueda recursiva de imágenes
        for ext in valid_extensions:
            # glob recursivo requiere ** y recursive=True
            found = glob.glob(os.path.join(directory_path, "**", ext), recursive=True)
            files.extend(found)
            
        print(f"Encontradas {len(files)} imágenes.")
        
        for i, file_path in enumerate(files):
            try:
                # Mostrar progreso
                print(f"Procesando [{i+1}/{len(files)}]: {os.path.basename(file_path)}...", end="\r")
                
                # Usar el detector existente para extraer métricas
                metrics = self.detector.analyze_image(file_path, generate_plots=False)
                
                if "error" not in metrics:
                    result = {
                        "filename": os.path.basename(file_path),
                        "path": file_path,
                        "label": label,
                        "laplacian_score": metrics.get("laplacian_score", 0),
                        "fft_score": metrics.get("fft_score", 0),
                        # Se pueden agregar más métricas si el detector las expone
                    }
                    self.results.append(result)
            except Exception as e:
                print(f"\nError procesando {file_path}: {e}")
        print(f"\nProcesamiento de {directory_path} completado.\n")

    def generate_report(self):
        """Genera un reporte estadístico y sugiere umbrales"""
        if not self.results:
            print("No hay resultados para analizar.")
            return

        # Separar por etiquetas
        labeled_data = {}
        for r in self.results:
            lbl = r.get("label", "unknown")
            if lbl not in labeled_data:
                labeled_data[lbl] = []
            labeled_data[lbl].append(r)

        print("\n" + "="*60)
        print("REPORTE DE ANÁLISIS DE CARACTERÍSTICAS (DETECCIÓN V2)")
        print("="*60)

        stats = {}

        for label, data in labeled_data.items():
            lap_scores = [d['laplacian_score'] for d in data]
            fft_scores = [d['fft_score'] for d in data]
            
            if not lap_scores: continue

            print(f"\nGrupo: {label.upper()} ({len(data)} imágenes)")
            print(f"  Laplacian - Media: {np.mean(lap_scores):.2f}, Std: {np.std(lap_scores):.2f}, Min: {np.min(lap_scores):.2f}, Max: {np.max(lap_scores):.2f}")
            print(f"  FFT       - Media: {np.mean(fft_scores):.2f}, Std: {np.std(fft_scores):.2f}, Min: {np.min(fft_scores):.2f}, Max: {np.max(fft_scores):.2f}")
            
            stats[label] = {
                "lap_mean": np.mean(lap_scores),
                "fft_mean": np.mean(fft_scores)
            }

        # Lógica para sugerir umbrales
        # Buscamos claves que parezcan 'real' y 'fake'
        real_keys = [k for k in stats.keys() if 'real' in k.lower()]
        fake_keys = [k for k in stats.keys() if 'fake' in k.lower() or 'ia' in k.lower() or 'gpt' in k.lower() or 'banana' in k.lower() or 'gen' in k.lower()]

        if real_keys and fake_keys:
            print("\n" + "-"*40)
            print("SUGERENCIA DE UMBRALES AUTOMÁTICA")
            print("-"*40)
            
            # Promediar todos los grupos identificados como reales y fakes
            avg_real_lap = np.mean([stats[k]['lap_mean'] for k in real_keys])
            avg_fake_lap = np.mean([stats[k]['lap_mean'] for k in fake_keys])
            
            avg_real_fft = np.mean([stats[k]['fft_mean'] for k in real_keys])
            avg_fake_fft = np.mean([stats[k]['fft_mean'] for k in fake_keys])
            
            # Punto medio simple
            suggested_lap_thresh = (avg_real_lap + avg_fake_lap) / 2
            suggested_fft_thresh = (avg_real_fft + avg_fake_fft) / 2
            
            print(f"Umbral Laplaciano sugerido: {suggested_lap_thresh:.2f}")
            print(f"  (Real avg: {avg_real_lap:.2f} vs Fake avg: {avg_fake_lap:.2f})")
            if avg_real_lap > avg_fake_lap:
                print("  Tendencia: Imágenes REALES tienen mayor detalle (score más alto).")
            else:
                print("  Tendencia: Imágenes REALES tienen menor score (más suaves?).")
            
            print(f"\nUmbral FFT sugerido: {suggested_fft_thresh:.2f}")
            print(f"  (Real avg: {avg_real_fft:.2f} vs Fake avg: {avg_fake_fft:.2f})")
            
            # Guardar configuración sugerida
            config = {
                "laplacian_threshold": float(suggested_lap_thresh),
                "fft_threshold": float(suggested_fft_thresh),
                "direction_laplacian": "lower_is_fake" if avg_real_lap > avg_fake_lap else "higher_is_fake",
                "direction_fft": "lower_is_fake" if avg_real_fft > avg_fake_fft else "higher_is_fake"
            }
            
            with open("suggested_config.json", "w") as f:
                json.dump(config, f, indent=4)
            print("\nConfiguración sugerida guardada en 'suggested_config.json'")
        else:
            print("\nNo se pudieron identificar claramente grupos 'real' y 'fake' para sugerir umbrales.")
            print("Asegúrate de usar etiquetas como 'real', 'fake', 'ia', 'gpt' en tus carpetas.")

if __name__ == "__main__":
    # Cambiar al directorio raíz del proyecto
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(PROJECT_ROOT)
    
    analyzer = AutoDetectorV2()
    
    print("Iniciando Detección V2 - Análisis Generalizado")
    print("Escaneando carpeta 'Imagenes'...")
    
    base_path = "Imagenes"
    if os.path.exists(base_path):
        # Listar subdirectorios
        items = os.listdir(base_path)
        for item in items:
            full_path = os.path.join(base_path, item)
            if os.path.isdir(full_path):
                # Heurística simple para etiquetar automáticamente basada en el nombre de la carpeta
                label = "unknown"
                name_lower = item.lower()
                
                if "real" in name_lower or "javi" in name_lower or "original" in name_lower:
                    label = "real"
                elif "ia" in name_lower or "gpt" in name_lower or "banana" in name_lower or "fake" in name_lower or "ejemplos" in name_lower:
                    label = "fake"
                
                analyzer.analyze_directory(full_path, label=label)
    else:
        print(f"No se encontró la carpeta '{base_path}'. Por favor edita el script para apuntar a tus carpetas.")

    analyzer.generate_report()
