"""
Detector Multi-Algoritmo para Imágenes Generadas por IA

Sistema modular que permite cambiar entre diferentes algoritmos de detección:
- Laplaciano (original)
- ELA (Error Level Analysis)
- Combinado (votación ponderada)
- Auto (selección automática según formato de imagen)

Cada algoritmo puede ser seleccionado dinámicamente desde la interfaz web.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import sys
import os
import imghdr
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from enum import Enum


# Formatos JPEG (ELA funciona mejor con estos)
JPEG_EXTENSIONS = {'.jpg', '.jpeg', '.jpe', '.jfif'}
JPEG_MIMETYPES = {'jpeg', 'jpg'}


def detect_image_format(image_path: str) -> dict:
    """
    Detecta el formato de una imagen.
    
    Args:
        image_path: Ruta al archivo de imagen
        
    Returns:
        dict con keys:
            - extension: extensión del archivo
            - detected_type: tipo detectado por imghdr (contenido real)
            - is_jpeg: True si es JPEG (por extensión o contenido)
            - recommended_algorithm: algoritmo recomendado según formato
    """
    path = Path(image_path)
    extension = path.suffix.lower()
    
    # Detectar tipo real por contenido (magic bytes)
    try:
        detected_type = imghdr.what(image_path)
    except Exception:
        detected_type = None
    
    # Determinar si es JPEG (por extensión O contenido)
    is_jpeg_by_ext = extension in JPEG_EXTENSIONS
    is_jpeg_by_content = detected_type in JPEG_MIMETYPES if detected_type else False
    is_jpeg = is_jpeg_by_ext or is_jpeg_by_content
    
    # Recomendar algoritmo según formato
    if is_jpeg:
        recommended = "ela"  # ELA funciona mejor con JPEG (84.38% accuracy)
        reason = "JPEG detectado - ELA es optimo para analizar artefactos de compresion"
    else:
        recommended = "laplacian"  # Laplacian es más general para otros formatos
        reason = f"Formato {detected_type or extension} - Laplaciano es mas robusto para formatos sin compresion JPEG"
    
    return {
        "extension": extension,
        "detected_type": detected_type,
        "is_jpeg": is_jpeg,
        "recommended_algorithm": recommended,
        "recommendation_reason": reason
    }


class DetectionAlgorithm(Enum):
    """Algoritmos de detección disponibles"""
    LAPLACIAN = "laplacian"
    ELA = "ela"
    COMBINED = "combined"
    AUTO = "auto"


# Configuración de umbrales por algoritmo
# Actualizado: 2025-12-02 - Calibración con dataset_features_combined.csv
ALGORITHM_CONFIG = {
    "laplacian": {
        "name": "Laplaciano",
        "description": "Análisis de alta frecuencia mediante filtro Laplaciano 3x3",
        "feature": "laplacian_score",
        "threshold": 4.6352,  # Calibrado sobre todas las imágenes
        "direction": "higher_is_fake",
        "accuracy": 68.52,  # Accuracy sobre todas (54 imgs)
        "accuracy_png": 95.45,  # Accuracy solo PNG (22 imgs)
        "unit": "Unidad 3: Filtros de vecindad",
        "best_for": ["png", "webp", "bmp", "tiff"]
    },
    "ela": {
        "name": "ELA (Error Level Analysis)",
        "description": "Análisis de niveles de error por recompresión JPEG",
        "feature": "ela_mean",
        "threshold": 0.1829,  # Calibrado sobre JPEG
        "direction": "lower_is_fake",
        "accuracy": 84.38,  # Accuracy solo JPEG (32 imgs)
        "unit": "Unidad 3: Operaciones aritméticas",
        "best_for": ["jpeg", "jpg"]
    },
    "combined": {
        "name": "Combinado (Votación)",
        "description": "Combinación ponderada de múltiples métricas",
        "feature": "combined_vote",
        "threshold": 0.5,
        "direction": "higher_is_fake",
        "accuracy": None,  # Depende de calibración
        "unit": "Múltiples unidades",
        "best_for": ["all"]
    },
    "auto": {
        "name": "Automático (según formato)",
        "description": "JPEG→ELA (84.38%), PNG→Laplaciano (95.45%)",
        "feature": "auto_selected",
        "threshold": None,  # Usa el threshold del algoritmo seleccionado
        "direction": "auto",
        "accuracy": 88.89,  # Accuracy combinada modo auto
        "unit": "Selección inteligente",
        "best_for": ["all"]
    }
}


class MultiAlgorithmDetector:
    """
    Detector que soporta múltiples algoritmos de detección.
    Permite cambiar dinámicamente el algoritmo desde la interfaz web.
    """
    
    def __init__(self, default_algorithm: str = "ela"):
        """
        Args:
            default_algorithm: Algoritmo por defecto ('laplacian', 'ela', 'combined')
        """
        self.current_algorithm = default_algorithm
        self.kernel_lap = np.array([[0, -1, 0],
                                     [-1, 4, -1],
                                     [0, -1, 0]])
        self._ela_analyzer = None
    
    def _get_ela_analyzer(self):
        """Importación diferida del analizador ELA."""
        if self._ela_analyzer is None:
            try:
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from detection_pipelinev2.analyzers.ela import ELAAnalyzer
                self._ela_analyzer = ELAAnalyzer(quality=90)
            except ImportError as e:
                print(f"Advertencia: No se pudo importar ELAAnalyzer: {e}")
        return self._ela_analyzer
    
    def get_available_algorithms(self) -> Dict[str, Dict]:
        """Retorna información sobre algoritmos disponibles."""
        return ALGORITHM_CONFIG.copy()
    
    def set_algorithm(self, algorithm: str):
        """
        Cambia el algoritmo de detección activo.
        
        Args:
            algorithm: Nombre del algoritmo ('laplacian', 'ela', 'combined')
        """
        if algorithm not in ALGORITHM_CONFIG:
            raise ValueError(f"Algoritmo no válido: {algorithm}. Disponibles: {list(ALGORITHM_CONFIG.keys())}")
        self.current_algorithm = algorithm
    
    def analyze_image(self, image_path: str, 
                      algorithm: Optional[str] = None,
                      generate_plots: bool = True) -> Dict[str, Any]:
        """
        Analiza una imagen ejecutando SIEMPRE ambos algoritmos (ELA y Laplaciano).
        Retorna el resultado con mayor confianza como primario.
        
        Args:
            image_path: Ruta a la imagen
            algorithm: Ignorado - siempre ejecuta ambos algoritmos
            generate_plots: Generar visualizaciones
            
        Returns:
            Diccionario con resultados primario y secundario del análisis
        """
        # Información de formato para discriminación y warnings
        format_info = detect_image_format(image_path)
        is_jpeg = format_info["is_jpeg"]
        
        # Cargar imagen
        img = self._load_image(image_path)
        if img is None:
            return {"error": "No se pudo cargar la imagen"}
        
        # Redimensionar si es muy grande
        img = self._resize_if_needed(img, max_dimension=2048)
        
        # Convertir formatos
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extraer métricas base (siempre se calculan para visualización)
        base_metrics = self._extract_base_metrics(img, img_gray, generate_plots)
        
        # Extraer métricas ELA (siempre, pero con warning para PNG)
        ela_metrics, ela_plots = self._extract_ela_metrics(img, generate_plots)
        
        # Clasificar con AMBOS algoritmos
        lap_prediction, lap_confidence, lap_details = self._classify(
            "laplacian", base_metrics, ela_metrics
        )
        ela_prediction, ela_confidence, ela_details = self._classify(
            "ela", base_metrics, ela_metrics
        )
        
        # Construir resultados de cada algoritmo
        laplacian_result = {
            "algorithm": "laplacian",
            "algorithm_name": ALGORITHM_CONFIG["laplacian"]["name"],
            "prediction": lap_prediction,
            "confidence": lap_confidence,
            "details": lap_details,
            "score": base_metrics.get("laplacian_score", 0),
            "threshold": ALGORITHM_CONFIG["laplacian"]["threshold"]
        }
        
        ela_result = {
            "algorithm": "ela",
            "algorithm_name": ALGORITHM_CONFIG["ela"]["name"],
            "prediction": ela_prediction,
            "confidence": ela_confidence,
            "details": ela_details,
            "score": ela_metrics.get("ela_mean", 0),
            "threshold": ALGORITHM_CONFIG["ela"]["threshold"],
            "reliable": is_jpeg  # ELA solo es confiable para JPEG
        }
        
        # Determinar resultado primario y secundario por confianza efectiva
        # - Si es JPEG: ELA tiene prioridad (bonificación de 15% a su confianza efectiva)
        # - Si no es JPEG: Penalizar ELA para que Laplaciano sea primario
        if is_jpeg:
            # Para JPEG, ELA es el algoritmo óptimo - darle ventaja
            effective_ela_conf = ela_confidence * 1.15  # Bonificación 15%
            effective_lap_conf = lap_confidence
        else:
            # Para PNG/otros, Laplaciano es más confiable
            effective_ela_conf = ela_confidence * 0.5   # Penalización 50%
            effective_lap_conf = lap_confidence
        
        if effective_ela_conf >= effective_lap_conf:
            primary_result = ela_result
            secondary_result = laplacian_result
        else:
            primary_result = laplacian_result
            secondary_result = ela_result
        
        # Construir respuesta final
        result = {
            "prediction": primary_result["prediction"],
            "confidence": primary_result["confidence"],
            "algorithm_used": primary_result["algorithm"],
            "algorithm_info": ALGORITHM_CONFIG[primary_result["algorithm"]],
            "algorithm_details": primary_result["details"],
            
            # Resultado primario detallado
            "primary_result": primary_result,
            
            # Resultado secundario
            "secondary_result": secondary_result,
            
            # Métricas generales
            "laplacian_score": round(base_metrics.get("laplacian_score", 0), 2),
            "fft_score": round(base_metrics.get("fft_score", 0), 2),
            "fft_metrics": base_metrics.get("fft_metrics", {}),
            "dimensions": img_rgb.shape,
            
            # Plots
            "histogram_plot": base_metrics.get("histogram_plot"),
            "laplacian_plot": base_metrics.get("laplacian_plot"),
            "fft_plot": base_metrics.get("fft_plot"),
            
            # Información de formato
            "format_info": format_info,
            
            # Warning para ELA en PNG
            "ela_warning": not is_jpeg
        }
        
        # Agregar métricas y plots ELA
        if ela_metrics:
            result["ela_metrics"] = ela_metrics
            result["ela_mean"] = round(ela_metrics.get("ela_mean", 0), 4)
        if ela_plots:
            result["ela_plot"] = ela_plots.get("ela_comparison")
            result["ela_heatmap"] = ela_plots.get("ela_heatmap")
        
        return result
    
    def _load_image(self, path: str) -> Optional[np.ndarray]:
        """Carga imagen soportando rutas con caracteres especiales."""
        try:
            img = cv2.imread(path)
            if img is None:
                with open(path, "rb") as f:
                    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None
    
    def _resize_if_needed(self, img: np.ndarray, max_dimension: int) -> np.ndarray:
        """Redimensiona imagen si excede tamaño máximo."""
        h, w = img.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img
    
    def _extract_base_metrics(self, img: np.ndarray, img_gray: np.ndarray, 
                               generate_plots: bool) -> Dict[str, Any]:
        """Extrae métricas base: histograma, laplaciano, FFT."""
        metrics = {}
        
        # Histograma
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        histogram = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
        if generate_plots:
            metrics["histogram_plot"] = self._plot_histogram(histogram)
        
        # Laplaciano
        laplacian = cv2.filter2D(img_gray, -1, self.kernel_lap)
        metrics["laplacian_score"] = float(np.mean(np.abs(laplacian)))
        if generate_plots:
            metrics["laplacian_plot"] = self._plot_laplacian(laplacian)
        
        # FFT
        fft_metrics, fft_plot = self._analyze_fft(img_gray, generate_plots)
        metrics["fft_score"] = fft_metrics.get("high_freq_energy", 0)
        metrics["fft_metrics"] = fft_metrics
        if generate_plots:
            metrics["fft_plot"] = fft_plot
        
        return metrics
    
    def _extract_ela_metrics(self, img: np.ndarray, 
                              generate_plots: bool) -> Tuple[Dict, Dict]:
        """Extrae métricas ELA."""
        ela = self._get_ela_analyzer()
        if ela is None:
            return {}, {}
        
        metrics = ela.analyze(img)
        plots = {}
        
        if generate_plots:
            comparison = ela.generate_comparison(img)
            heatmap = ela.generate_visualization(img)
            plots["ela_comparison"] = self._image_to_base64(comparison)
            plots["ela_heatmap"] = self._image_to_base64(heatmap)
        
        return metrics, plots
    
    def _classify(self, algorithm: str, 
                  base_metrics: Dict, 
                  ela_metrics: Dict) -> Tuple[str, float, Dict]:
        """
        Clasifica imagen según algoritmo seleccionado.
        
        Returns:
            (predicción, confianza, detalles)
        """
        config = ALGORITHM_CONFIG[algorithm]
        details = {"algorithm": algorithm}
        
        if algorithm == "laplacian":
            value = base_metrics.get("laplacian_score", 0)
            threshold = config["threshold"]
            is_fake = value > threshold if config["direction"] == "higher_is_fake" else value < threshold
            
            distance = abs(value - threshold)
            confidence = min(95.0, 60.0 + distance * 10)
            
            details["value"] = round(value, 4)
            details["threshold"] = threshold
            
        elif algorithm == "ela":
            value = ela_metrics.get("ela_mean", 0)
            threshold = config["threshold"]
            is_fake = value < threshold if config["direction"] == "lower_is_fake" else value > threshold
            
            distance = abs(value - threshold)
            max_dist = threshold * 2
            confidence = min(95.0, 60.0 + (distance / max_dist) * 35.0)
            
            details["value"] = round(value, 4)
            details["threshold"] = threshold
            
        elif algorithm == "combined":
            # Votación ponderada
            votes = []
            
            # Voto Laplaciano
            lap_val = base_metrics.get("laplacian_score", 0)
            lap_thresh = ALGORITHM_CONFIG["laplacian"]["threshold"]
            lap_vote = 1 if lap_val > lap_thresh else 0
            lap_conf = min(95.0, 60.0 + abs(lap_val - lap_thresh) * 10)
            votes.append(("laplacian", lap_vote, lap_conf, 1.0))
            
            # Voto ELA (mayor peso por mejor accuracy)
            if ela_metrics:
                ela_val = ela_metrics.get("ela_mean", 0)
                ela_thresh = ALGORITHM_CONFIG["ela"]["threshold"]
                ela_vote = 1 if ela_val < ela_thresh else 0
                ela_conf = min(95.0, 60.0 + abs(ela_val - ela_thresh) * 50)
                votes.append(("ela", ela_vote, ela_conf, 1.5))  # Mayor peso
            
            # Calcular votación ponderada
            total_weight = sum(v[3] * v[2] for v in votes)
            weighted_sum = sum(v[1] * v[2] * v[3] for v in votes)
            combined_vote = weighted_sum / total_weight if total_weight > 0 else 0.5
            
            is_fake = combined_vote > 0.5
            confidence = 60.0 + abs(combined_vote - 0.5) * 70.0
            
            details["votes"] = {v[0]: {"vote": v[1], "conf": round(v[2], 2)} for v in votes}
            details["combined_vote"] = round(combined_vote, 3)
            value = combined_vote
        
        prediction = "GENERADA POR IA" if is_fake else "REAL"
        
        return prediction, round(confidence, 2), details
    
    def _plot_histogram(self, histogram: np.ndarray) -> str:
        """Genera gráfico del histograma en base64."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(histogram)
        ax.set_title("Histograma de la imagen")
        ax.set_xlabel("Valor de Píxel")
        ax.set_ylabel("Frecuencia")
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return f"data:image/png;base64,{image_base64}"
    
    def _plot_laplacian(self, laplacian: np.ndarray) -> str:
        """Genera visualización del mapa Laplaciano."""
        if max(laplacian.shape) > 800:
            scale = 800 / max(laplacian.shape)
            new_size = (int(laplacian.shape[1] * scale), int(laplacian.shape[0] * scale))
            laplacian = cv2.resize(laplacian, new_size, interpolation=cv2.INTER_AREA)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(laplacian, cmap='gray')
        ax.set_title("Mapa Laplaciano")
        ax.axis('off')
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return f"data:image/png;base64,{image_base64}"
    
    def _analyze_fft(self, img_gray: np.ndarray, 
                     generate_plot: bool = True) -> Tuple[Dict, Optional[str]]:
        """Análisis espectral usando FFT 2D."""
        max_size = max(img_gray.shape)
        scale_factor = 0.05 if max_size > 2000 else 0.1
        
        img_small = cv2.resize(img_gray, None, fx=scale_factor, fy=scale_factor,
                               interpolation=cv2.INTER_AREA)
        
        f = np.fft.fft2(img_small)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        mask_size = min(center_y, center_x) // 4
        
        mask = np.ones(magnitude_spectrum.shape, dtype=bool)
        mask[center_y - mask_size:center_y + mask_size,
             center_x - mask_size:center_x + mask_size] = False
        
        high_freq_energy = float(np.mean(magnitude_spectrum[mask]))
        
        # Métricas adicionales
        magnitud = np.abs(fshift)
        h, w = magnitud.shape
        cy, cx = h // 2, w // 2
        
        radio_bajo = min(h, w) * 0.15
        y, x = np.ogrid[:h, :w]
        dist_sq = (x - cx)**2 + (y - cy)**2
        mascara_bajo = dist_sq <= radio_bajo**2
        
        energia_total = np.sum(magnitud**2)
        energia_baja = np.sum(magnitud[mascara_bajo]**2)
        energia_alta = energia_total - energia_baja
        ratio_af_bf = energia_alta / energia_baja if energia_baja > 0 else 0
        
        metrics = {
            'high_freq_energy': high_freq_energy,
            'ratio_af_bf': ratio_af_bf
        }
        
        fft_plot = None
        if generate_plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(magnitude_spectrum, cmap='gray')
            ax.set_title("Espectro de Frecuencias (FFT 2D)")
            ax.axis('off')
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            fft_plot = f"data:image/png;base64,{image_base64}"
        
        return metrics, fft_plot
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convierte imagen OpenCV a string base64."""
        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"


# Instancia global para mantener estado
_detector_instance = None

def get_detector() -> MultiAlgorithmDetector:
    """Obtiene instancia singleton del detector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MultiAlgorithmDetector(default_algorithm="ela")
    return _detector_instance
