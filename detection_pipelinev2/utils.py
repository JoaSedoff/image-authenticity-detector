"""
Utilidades compartidas para pipeline de detección forense.
Funciones de carga, normalización y procesamiento común.
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional, List, Dict


def load_image(path: str, max_size: int = 2048) -> Optional[np.ndarray]:
    """
    Carga imagen desde archivo con redimensionado opcional.
    Soporta rutas con caracteres especiales (acentos en Windows).
    
    Args:
        path: Ruta al archivo de imagen
        max_size: Tamaño máximo del lado mayor (para optimizar memoria)
        
    Returns:
        Imagen BGR o None si falla la carga
    """
    if not os.path.exists(path):
        return None
    
    # Intentar carga normal primero
    image = cv2.imread(path)
    
    # Si falla, usar método alternativo para rutas con acentos (Windows)
    if image is None:
        try:
            with open(path, "rb") as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception:
            return None
    
    if image is None:
        return None
    
    # Redimensionar si excede tamaño máximo
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image


def is_jpeg(path: str) -> bool:
    """
    Verifica si archivo es formato JPEG (por extensión).
    """
    ext = os.path.splitext(path)[1].lower()
    return ext in ['.jpg', '.jpeg']


def calculate_entropy(data: np.ndarray) -> float:
    """
    Calcula entropía de Shannon de una imagen o región.
    Unidad 2: Análisis de histograma
    
    Args:
        data: Imagen o array de valores
        
    Returns:
        Valor de entropía en bits
    """
    # Aplanar si es multidimensional
    flat = data.flatten().astype(np.uint8)
    
    # Calcular histograma normalizado
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    hist = hist[hist > 0]  # Eliminar bins vacíos
    
    # Normalizar a probabilidades
    probs = hist / hist.sum()
    
    # Entropía de Shannon
    entropy = -np.sum(probs * np.log2(probs))
    
    return entropy


def split_into_blocks(image: np.ndarray, block_size: int = 8) -> List[np.ndarray]:
    """
    Divide imagen en bloques cuadrados (para análisis JPEG).
    Unidad 4: Extracción de regiones
    
    Args:
        image: Imagen en escala de grises
        block_size: Tamaño del bloque (default 8x8 para JPEG)
        
    Returns:
        Lista de bloques como arrays
    """
    h, w = image.shape[:2]
    blocks = []
    
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = image[y:y+block_size, x:x+block_size]
            blocks.append(block)
    
    return blocks


def load_dataset_labels(data_path: str) -> Dict[str, int]:
    """
    Carga etiquetas desde archivo data.txt.
    Formato: nombre_archivo, etiqueta (0=real, 1=IA)
    
    Args:
        data_path: Ruta al archivo data.txt
        
    Returns:
        Diccionario {nombre_archivo: etiqueta}
    """
    labels = {}
    
    if not os.path.exists(data_path):
        return labels
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            
            parts = line.split(',')
            if len(parts) >= 2:
                filename = parts[0].strip()
                try:
                    label = int(parts[1].strip())
                    labels[filename] = label
                except ValueError:
                    continue
    
    return labels


def apply_colormap(image: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Aplica mapa de color a imagen en escala de grises.
    Útil para visualización de mapas de error.
    
    Args:
        image: Imagen en escala de grises
        colormap: Tipo de mapa de color OpenCV
        
    Returns:
        Imagen BGR con mapa de color aplicado
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalizar a 0-255
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return cv2.applyColorMap(normalized, colormap)
