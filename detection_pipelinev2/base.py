"""
Clase base abstracta para analizadores forenses.
Unidad 3: Filtros y transformaciones de vecindad
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple


class BaseAnalyzer(ABC):
    """
    Clase base para todos los analizadores de imágenes forenses.
    Define interfaz común para extracción de características.
    """
    
    def __init__(self, name: str):
        """
        Args:
            name: Identificador del analizador
        """
        self.name = name
        self._last_visualization = None
    
    @abstractmethod
    def analyze(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrae métricas de la imagen.
        
        Args:
            image: Imagen en formato BGR (OpenCV) o escala de grises
            
        Returns:
            Diccionario con métricas extraídas
        """
        pass
    
    @abstractmethod
    def generate_visualization(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Genera imagen de visualización del análisis.
        
        Args:
            image: Imagen original BGR
            
        Returns:
            Imagen de visualización o None si no aplica
        """
        pass
    
    def get_last_visualization(self) -> Optional[np.ndarray]:
        """Retorna la última visualización generada."""
        return self._last_visualization
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normaliza imagen para procesamiento consistente.
        Convierte a uint8 si es necesario.
        """
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return image
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convierte imagen a escala de grises.
        Unidad 2: Transformaciones de intensidad
        """
        import cv2
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
