"""
ELA (Error Level Analysis) - Análisis de Nivel de Error

Técnica forense para detectar manipulación/síntesis mediante análisis
de inconsistencias en niveles de compresión JPEG.

Fundamento teórico:
- Unidad 2: Histograma, análisis de intensidad
- Unidad 3: Operaciones aritméticas (diferencia absoluta)
- Unidad 4: Extracción de regiones, análisis de bloques 8x8

Principio: Al recomprimir una imagen JPEG, las regiones que ya fueron
comprimidas al mismo nivel muestran poco cambio, mientras que regiones
manipuladas o sintéticas presentan diferencias notables.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import tempfile
import os

from ..base import BaseAnalyzer
from ..utils import calculate_entropy, apply_colormap


class ELAAnalyzer(BaseAnalyzer):
    """
    Analizador ELA para detección de manipulación/síntesis.
    
    Extrae métricas basadas en diferencias de nivel de compresión
    entre imagen original y recomprimida.
    """
    
    def __init__(self, quality: int = 90, scale_factor: int = 15):
        """
        Args:
            quality: Nivel de calidad JPEG para recompresión (1-100)
            scale_factor: Factor de amplificación para visualización
        """
        super().__init__("ELA")
        self.quality = quality
        self.scale_factor = scale_factor
        self._ela_map = None
    
    def analyze(self, image: np.ndarray) -> Dict[str, float]:
        """
        Realiza análisis ELA sobre la imagen.
        
        Proceso:
        1. Guardar imagen temporalmente como JPEG
        2. Recargar imagen recomprimida
        3. Calcular diferencia absoluta (Unidad 3: ops aritméticas)
        4. Extraer métricas estadísticas del mapa de error
        
        Args:
            image: Imagen BGR
            
        Returns:
            Diccionario con métricas ELA:
            - ela_mean: Media del error (indicador principal)
            - ela_std: Desviación estándar del error
            - ela_max: Valor máximo de error
            - ela_entropy: Entropía del mapa de error
            - ela_high_error_ratio: Proporción de píxeles con alto error
            - ela_block_variance: Varianza entre bloques 8x8
        """
        image = self._normalize_image(image)
        
        # Generar mapa ELA
        ela_map = self._compute_ela_map(image)
        self._ela_map = ela_map
        
        # Convertir a escala de grises para métricas
        if len(ela_map.shape) == 3:
            ela_gray = cv2.cvtColor(ela_map, cv2.COLOR_BGR2GRAY)
        else:
            ela_gray = ela_map
        
        # Métricas básicas de intensidad (Unidad 2)
        ela_mean = float(np.mean(ela_gray))
        ela_std = float(np.std(ela_gray))
        ela_max = float(np.max(ela_gray))
        
        # Entropía del mapa de error (Unidad 2: histograma)
        ela_entropy = calculate_entropy(ela_gray)
        
        # Proporción de píxeles con alto error (umbral adaptativo)
        threshold = ela_mean + 2 * ela_std
        high_error_mask = ela_gray > threshold
        ela_high_error_ratio = float(np.sum(high_error_mask) / ela_gray.size)
        
        # Varianza entre bloques 8x8 (Unidad 4: análisis de regiones)
        block_variance = self._compute_block_variance(ela_gray, block_size=8)
        
        # Análisis de uniformidad espacial
        ela_uniformity = self._compute_spatial_uniformity(ela_gray)
        
        return {
            "ela_mean": round(ela_mean, 4),
            "ela_std": round(ela_std, 4),
            "ela_max": round(ela_max, 4),
            "ela_entropy": round(ela_entropy, 4),
            "ela_high_error_ratio": round(ela_high_error_ratio, 6),
            "ela_block_variance": round(block_variance, 4),
            "ela_uniformity": round(ela_uniformity, 4)
        }
    
    def _compute_ela_map(self, image: np.ndarray) -> np.ndarray:
        """
        Calcula mapa de Error Level Analysis.
        
        Unidad 3: Operaciones aritméticas - diferencia absoluta
        """
        # Crear archivo temporal para recompresión
        temp_path = None
        try:
            # Guardar como JPEG con calidad especificada
            fd, temp_path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
            
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            cv2.imwrite(temp_path, image, encode_params)
            
            # Recargar imagen recomprimida
            recompressed = cv2.imread(temp_path, cv2.IMREAD_COLOR)
            
            if recompressed is None:
                # Fallback si falla la lectura
                return np.zeros_like(image)
            
            # Asegurar mismo tamaño
            if recompressed.shape != image.shape:
                recompressed = cv2.resize(recompressed, 
                                         (image.shape[1], image.shape[0]))
            
            # Diferencia absoluta (Unidad 3: operaciones aritméticas)
            ela_map = cv2.absdiff(image, recompressed)
            
            return ela_map
            
        finally:
            # Limpiar archivo temporal
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def _compute_block_variance(self, ela_gray: np.ndarray, 
                                 block_size: int = 8) -> float:
        """
        Calcula varianza de energía entre bloques.
        
        Unidad 4: Extracción de regiones
        
        Imágenes manipuladas suelen tener alta varianza entre bloques
        debido a diferentes niveles de compresión en distintas regiones.
        """
        h, w = ela_gray.shape
        block_means = []
        
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = ela_gray[y:y+block_size, x:x+block_size]
                block_means.append(np.mean(block))
        
        if len(block_means) < 2:
            return 0.0
        
        return float(np.var(block_means))
    
    def _compute_spatial_uniformity(self, ela_gray: np.ndarray) -> float:
        """
        Mide uniformidad espacial del error.
        
        Divide imagen en cuadrantes y compara distribución.
        Imágenes reales tienden a tener error más uniforme.
        """
        h, w = ela_gray.shape
        mid_h, mid_w = h // 2, w // 2
        
        # Cuatro cuadrantes
        q1 = ela_gray[:mid_h, :mid_w]
        q2 = ela_gray[:mid_h, mid_w:]
        q3 = ela_gray[mid_h:, :mid_w]
        q4 = ela_gray[mid_h:, mid_w:]
        
        means = [np.mean(q1), np.mean(q2), np.mean(q3), np.mean(q4)]
        
        # Uniformidad = 1 - (coef. variación normalizado)
        if np.mean(means) > 0:
            cv = np.std(means) / np.mean(means)
            uniformity = 1.0 / (1.0 + cv)
        else:
            uniformity = 1.0
        
        return uniformity
    
    def generate_visualization(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Genera visualización del mapa ELA con mapa de calor.
        
        Args:
            image: Imagen original BGR
            
        Returns:
            Imagen con mapa ELA coloreado (jet colormap)
        """
        image = self._normalize_image(image)
        
        # Calcular mapa ELA si no existe
        if self._ela_map is None:
            self._ela_map = self._compute_ela_map(image)
        
        ela_map = self._ela_map
        
        # Amplificar para mejor visualización
        ela_amplified = cv2.multiply(ela_map, self.scale_factor)
        ela_amplified = np.clip(ela_amplified, 0, 255).astype(np.uint8)
        
        # Convertir a escala de grises
        if len(ela_amplified.shape) == 3:
            ela_gray = cv2.cvtColor(ela_amplified, cv2.COLOR_BGR2GRAY)
        else:
            ela_gray = ela_amplified
        
        # Aplicar mapa de calor (Jet: azul=bajo, rojo=alto)
        ela_colored = apply_colormap(ela_gray, cv2.COLORMAP_JET)
        
        self._last_visualization = ela_colored
        
        return ela_colored
    
    def generate_comparison(self, image: np.ndarray) -> np.ndarray:
        """
        Genera imagen comparativa: Original | Mapa ELA | Mapa de calor
        
        Útil para presentación académica.
        """
        image = self._normalize_image(image)
        
        # Mapa ELA
        ela_map = self._compute_ela_map(image)
        ela_amplified = cv2.multiply(ela_map, self.scale_factor)
        ela_amplified = np.clip(ela_amplified, 0, 255).astype(np.uint8)
        
        # Mapa de calor
        if len(ela_amplified.shape) == 3:
            ela_gray = cv2.cvtColor(ela_amplified, cv2.COLOR_BGR2GRAY)
        else:
            ela_gray = ela_amplified
        ela_heatmap = apply_colormap(ela_gray, cv2.COLORMAP_JET)
        
        # Redimensionar todas al mismo tamaño
        h, w = image.shape[:2]
        target_w = min(w, 400)  # Limitar ancho para visualización
        scale = target_w / w
        target_h = int(h * scale)
        
        img_resized = cv2.resize(image, (target_w, target_h))
        ela_resized = cv2.resize(ela_amplified, (target_w, target_h))
        heat_resized = cv2.resize(ela_heatmap, (target_w, target_h))
        
        # Concatenar horizontalmente
        comparison = np.hstack([img_resized, ela_resized, heat_resized])
        
        # Agregar etiquetas
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(comparison, "ELA", (target_w + 10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(comparison, "Heatmap", (2*target_w + 10, 25), font, 0.6, (255, 255, 255), 2)
        
        return comparison
