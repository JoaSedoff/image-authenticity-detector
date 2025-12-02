"""
Pipeline Forense V2

Orquestador que combina múltiples analizadores forenses:
- ELA (Error Level Analysis) 
- Métricas existentes de model/detector.py (Laplaciano, FFT)

Diseñado para coexistir con el pipeline anterior.
"""

import cv2
import numpy as np
import os
import sys
from typing import Dict, Optional, List, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from .analyzers.ela import ELAAnalyzer
from .utils import load_image, is_jpeg


class ForensicPipeline:
    """
    Pipeline de análisis forense que combina múltiples técnicas.
    
    Integra ELA con el detector existente para proporcionar
    un conjunto más completo de métricas discriminativas.
    """
    
    def __init__(self, ela_quality: int = 90):
        """
        Args:
            ela_quality: Calidad JPEG para análisis ELA (default 90)
        """
        self.ela_analyzer = ELAAnalyzer(quality=ela_quality)
        self._legacy_detector = None
    
    def _get_legacy_detector(self):
        """Importación diferida del detector existente para evitar ciclos."""
        if self._legacy_detector is None:
            try:
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from model.detector import AIImageDetector
                self._legacy_detector = AIImageDetector()
            except ImportError as e:
                print(f"Advertencia: No se pudo importar detector legacy: {e}")
        return self._legacy_detector
    
    def analyze(self, image_path: str, 
                include_legacy: bool = True,
                generate_plots: bool = True) -> Dict[str, Any]:
        """
        Análisis forense completo de una imagen.
        
        Args:
            image_path: Ruta a la imagen
            include_legacy: Incluir métricas del detector anterior (Laplaciano, FFT)
            generate_plots: Generar visualizaciones
            
        Returns:
            Diccionario con todas las métricas y visualizaciones
        """
        # Cargar imagen
        image = load_image(image_path)
        if image is None:
            return {"error": f"No se pudo cargar imagen: {image_path}"}
        
        result = {
            "filename": os.path.basename(image_path),
            "path": image_path,
            "is_jpeg": is_jpeg(image_path),
            "dimensions": image.shape
        }
        
        # Análisis ELA
        ela_metrics = self.ela_analyzer.analyze(image)
        result["ela"] = ela_metrics
        
        # Generar visualización ELA
        if generate_plots:
            ela_comparison = self.ela_analyzer.generate_comparison(image)
            result["ela_plot"] = self._image_to_base64(ela_comparison)
            
            ela_heatmap = self.ela_analyzer.generate_visualization(image)
            result["ela_heatmap"] = self._image_to_base64(ela_heatmap)
        
        # Incluir métricas del detector legacy si está disponible
        if include_legacy:
            legacy = self._get_legacy_detector()
            if legacy:
                legacy_result = legacy.analyze_image(image_path, generate_plots=generate_plots)
                if "error" not in legacy_result:
                    result["legacy"] = {
                        "laplacian_score": legacy_result.get("laplacian_score"),
                        "fft_score": legacy_result.get("fft_score"),
                        "fft_metrics": legacy_result.get("fft_metrics", {})
                    }
                    if generate_plots:
                        result["laplacian_plot"] = legacy_result.get("laplacian_plot")
                        result["fft_plot"] = legacy_result.get("fft_plot")
                        result["histogram_plot"] = legacy_result.get("histogram_plot")
        
        # Combinar métricas para clasificación
        result["combined_metrics"] = self._combine_metrics(result)
        
        return result
    
    def analyze_batch(self, image_paths: List[str], 
                      jpeg_only: bool = False) -> List[Dict[str, Any]]:
        """
        Analiza múltiples imágenes.
        
        Args:
            image_paths: Lista de rutas a imágenes
            jpeg_only: Si True, solo procesa archivos JPEG
            
        Returns:
            Lista de resultados de análisis
        """
        results = []
        
        for path in image_paths:
            if jpeg_only and not is_jpeg(path):
                continue
            
            result = self.analyze(path, generate_plots=False)
            results.append(result)
        
        return results
    
    def _combine_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """
        Combina métricas de todos los analizadores en diccionario plano.
        """
        combined = {}
        
        # Métricas ELA
        if "ela" in result:
            for key, value in result["ela"].items():
                combined[key] = value
        
        # Métricas legacy
        if "legacy" in result:
            legacy = result["legacy"]
            combined["laplacian_score"] = legacy.get("laplacian_score", 0)
            combined["fft_score"] = legacy.get("fft_score", 0)
            
            fft_metrics = legacy.get("fft_metrics", {})
            for key, value in fft_metrics.items():
                combined[f"fft_{key}"] = value
        
        return combined
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convierte imagen OpenCV a string base64 para web."""
        # Codificar como PNG
        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    def extract_features_for_calibration(self, image_path: str, 
                                         include_legacy: bool = False) -> Optional[Dict[str, float]]:
        """
        Extrae solo métricas numéricas para calibración.
        Sin visualizaciones para mayor velocidad.
        
        Args:
            image_path: Ruta a la imagen
            include_legacy: Incluir métricas del detector legacy (más lento)
            
        Returns:
            Diccionario con métricas o None si falla
        """
        result = self.analyze(image_path, include_legacy=include_legacy, generate_plots=False)
        
        if "error" in result:
            return None
        
        return result.get("combined_metrics", {})
