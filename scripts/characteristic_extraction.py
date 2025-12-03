"""
Script de Extracción de Características para Detección de Imágenes IA

Extrae métricas de:
1. Laplaciano (para todas las imágenes) - Análisis de alta frecuencia
2. ELA (solo JPEG) - Error Level Analysis para artefactos de compresión
3. Métricas espectrales FFT (para todas)
4. Histograma/Intensidad (para todas)

Las PNG se ignoran para métricas ELA ya que ELA funciona mejor con JPEG.
"""

import cv2
import numpy as np
import pandas as pd
import os
import sys

# Agregar directorio raíz del proyecto al path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Configuración de rutas
BASE_DIR = os.path.join('Imagenes', 'DeteccionIA')
DATA_FILE = os.path.join(BASE_DIR, 'data.txt')
OUTPUT_FILE = 'dataset_features_combined.csv'

# Extensiones JPEG
JPEG_EXTENSIONS = {'.jpg', '.jpeg', '.jpe', '.jfif'}


def is_jpeg(filename: str) -> bool:
    """Verifica si el archivo es JPEG por extensión."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in JPEG_EXTENSIONS


def load_image(img_path: str):
    """Carga imagen de forma segura para Windows con caracteres especiales."""
    try:
        with open(img_path, "rb") as stream:
            file_bytes = bytearray(stream.read())
            numpyarray = np.asarray(file_bytes, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error cargando imagen: {e}")
        return None


def safe_process_image(img, img_bgr, is_jpeg_file, filename):
    """
    Procesa una imagen de forma segura, capturando errores de JPEG corrupto.
    Retorna dict de features o None si falla.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = {}
        
        # Métricas para TODAS las imágenes
        features.update(get_histogram_metrics(gray))
        features.update(get_laplacian_metrics(gray))
        features.update(get_spectral_metrics(gray))
        
        # Métricas ELA SOLO para JPEG
        if is_jpeg_file:
            try:
                features.update(get_ela_metrics(img_bgr))
            except Exception as e:
                print(f"  Advertencia ELA para {filename}: {e}")
                features.update({
                    'ela_mean': np.nan,
                    'ela_std': np.nan,
                    'ela_max': np.nan,
                    'ela_entropy': np.nan,
                    'ela_high_error_ratio': np.nan,
                    'ela_block_variance': np.nan,
                    'ela_uniformity': np.nan
                })
        else:
            features.update({
                'ela_mean': np.nan,
                'ela_std': np.nan,
                'ela_max': np.nan,
                'ela_entropy': np.nan,
                'ela_high_error_ratio': np.nan,
                'ela_block_variance': np.nan,
                'ela_uniformity': np.nan
            })
        
        return features
    except Exception as e:
        print(f"  Error procesando {filename}: {e}")
        return None


# =============================================================================
# MÉTRICAS DE HISTOGRAMA/INTENSIDAD
# =============================================================================
def get_histogram_metrics(gray_img):
    """Métricas estadísticas basadas en histograma e intensidad."""
    mean_val = np.mean(gray_img)
    std_val = np.std(gray_img)
    
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    hist_norm = hist_norm[hist_norm > 0]
    entropy = -np.sum(hist_norm * np.log2(hist_norm))
    
    return {
        'intensity_mean': mean_val,
        'intensity_std': std_val,
        'intensity_entropy': entropy
    }


# =============================================================================
# MÉTRICAS LAPLACIANAS (Alta Frecuencia / Bordes)
# =============================================================================
def get_laplacian_metrics(gray_img):
    """
    Métricas de alta frecuencia usando filtro Laplaciano 3x3.
    Threshold calibrado: 4.5853, higher_is_fake, 68.52% accuracy
    """
    kernel_lap = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])
    
    lap_img = cv2.filter2D(gray_img, -1, kernel_lap)
    
    return {
        'laplacian_mean': np.mean(np.abs(lap_img)),
        'laplacian_var': np.var(lap_img),
        'laplacian_max': np.max(np.abs(lap_img)),
        'laplacian_std': np.std(lap_img)
    }

# =============================================================================
# MÉTRICAS ESPECTRALES (FFT 2D)
# =============================================================================
def get_spectral_metrics(gray_img):
    """Métricas en el dominio de la frecuencia (FFT 2D)."""
    h, w = gray_img.shape
    target_size = 512
    scale = min(1.0, target_size / max(h, w))
    
    if scale < 1.0:
        img_small = cv2.resize(gray_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img_small = gray_img

    f = np.fft.fft2(img_small)
    fshift = np.fft.fftshift(f)
    magnitud = np.abs(fshift)
    
    h_fft, w_fft = magnitud.shape
    cy, cx = h_fft // 2, w_fft // 2
    y, x = np.ogrid[:h_fft, :w_fft]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_radio = min(h_fft, w_fft) // 2
    
    radio_bajo = min(h_fft, w_fft) * 0.15
    mascara_bajo = dist <= radio_bajo
    
    energia_total = np.sum(magnitud**2)
    energia_baja = np.sum(magnitud[mascara_bajo]**2)
    energia_alta = energia_total - energia_baja
    
    ratio_af_bf = energia_alta / energia_baja if energia_baja > 0 else 0
    
    num_anillos = 20
    energias_anillos = []
    for i in range(num_anillos):
        r_inner = (i / num_anillos) * max_radio
        r_outer = ((i + 1) / num_anillos) * max_radio
        mascara_anillo = (dist >= r_inner) & (dist < r_outer)
        if np.sum(mascara_anillo) > 0:
            energias_anillos.append(np.mean(magnitud[mascara_anillo]))
            
    uniformidad_radial = np.std(energias_anillos) / (np.mean(energias_anillos) + 1e-10)
    
    magnitud_norm = magnitud / (np.sum(magnitud) + 1e-10)
    magnitud_norm = magnitud_norm[magnitud_norm > 0]
    fft_entropia = -np.sum(magnitud_norm * np.log(magnitud_norm + 1e-10))
    
    return {
        'fft_ratio_af_bf': ratio_af_bf,
        'fft_uniformidad': uniformidad_radial,
        'fft_entropia': fft_entropia
    }


# =============================================================================
# MÉTRICAS ELA (Error Level Analysis) - SOLO JPEG
# =============================================================================
def get_ela_metrics(img_bgr, quality: int = 90):
    """
    Métricas ELA (Error Level Analysis) - SOLO para JPEG.
    
    Threshold calibrado: 0.1839, lower_is_fake, 84.38% accuracy
    """
    # Recomprimir como JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img_bgr, encode_param)
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    # Mapa ELA = diferencia absoluta
    ela_map = cv2.absdiff(img_bgr, recompressed)
    ela_gray = cv2.cvtColor(ela_map, cv2.COLOR_BGR2GRAY)
    
    # Métricas básicas
    ela_mean = float(np.mean(ela_gray))
    ela_std = float(np.std(ela_gray))
    ela_max = float(np.max(ela_gray))
    
    # Entropía
    hist = cv2.calcHist([ela_gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    hist_norm = hist_norm[hist_norm > 0]
    ela_entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
    
    # Proporción de alto error
    threshold = ela_mean + 2 * ela_std
    high_error_mask = ela_gray > threshold
    ela_high_error_ratio = float(np.sum(high_error_mask) / ela_gray.size)
    
    # Varianza entre bloques 8x8
    h, w = ela_gray.shape
    block_size = 8
    block_means = []
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = ela_gray[i:i+block_size, j:j+block_size]
            block_means.append(np.mean(block))
    ela_block_variance = np.var(block_means) if block_means else 0
    
    # Uniformidad espacial
    h_half, w_half = h // 2, w // 2
    quadrants = [
        ela_gray[:h_half, :w_half],
        ela_gray[:h_half, w_half:],
        ela_gray[h_half:, :w_half],
        ela_gray[h_half:, w_half:]
    ]
    quadrant_means = [np.mean(q) for q in quadrants]
    ela_uniformity = 1.0 - (np.std(quadrant_means) / (np.mean(quadrant_means) + 1e-10))
    
    return {
        'ela_mean': round(ela_mean, 4),
        'ela_std': round(ela_std, 4),
        'ela_max': round(ela_max, 4),
        'ela_entropy': round(ela_entropy, 4),
        'ela_high_error_ratio': round(ela_high_error_ratio, 6),
        'ela_block_variance': round(ela_block_variance, 4),
        'ela_uniformity': round(ela_uniformity, 4)
    }


# =============================================================================
# PROCESAMIENTO PRINCIPAL
# =============================================================================
def process_dataset():
    print("=" * 60)
    print("EXTRACCIÓN DE CARACTERÍSTICAS - LAPLACIANO + ELA")
    print("=" * 60)
    print(f"Leyendo etiquetas de: {DATA_FILE}")
    print(f"Nota: Métricas ELA solo se calculan para imágenes JPEG")
    print()
    
    results = []
    
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: No se encontró el archivo de datos: {DATA_FILE}")
        return

    with open(DATA_FILE, 'r') as f:
        lines = f.readlines()

    count = 0
    jpeg_count = 0
    png_count = 0
    
    for line in lines:
        line = line.strip()
        if not line: 
            continue
        
        parts = line.split(',')
        if len(parts) < 2: 
            continue
        
        filename = parts[0].strip()
        try:
            label = int(parts[1].strip())
        except ValueError:
            continue
        
        img_path = os.path.join(BASE_DIR, filename)
        
        if not os.path.exists(img_path):
            print(f"Advertencia: Imagen no encontrada {img_path}")
            continue
            
        img = load_image(img_path)
        if img is None:
            print(f"  Saltando (no se pudo cargar): {filename}")
            continue
        
        is_jpeg_file = is_jpeg(filename)
        if is_jpeg_file:
            jpeg_count += 1
        else:
            png_count += 1
        
        # Procesar imagen de forma segura
        features = safe_process_image(img, img, is_jpeg_file, filename)
        if features is None:
            print(f"  Saltando (error en procesamiento): {filename}")
            continue
        
        # Agregar metadatos
        features['filename'] = filename
        features['label'] = label
        features['is_jpeg'] = is_jpeg_file
        
        results.append(features)
        count += 1
        if count % 10 == 0:
            print(f"Procesadas {count} imágenes...")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)
        
        print()
        print("=" * 60)
        print("PROCESAMIENTO COMPLETADO")
        print("=" * 60)
        print(f"Total: {len(df)} | JPEG: {jpeg_count} | PNG: {png_count}")
        print(f"Guardado en: {OUTPUT_FILE}")
        
        # Resumen Laplaciano
        print("\n--- LAPLACIANO (todas) ---")
        for lbl in [0, 1]:
            data = df[df['label'] == lbl]['laplacian_mean']
            print(f"  Label {lbl}: mean={data.mean():.4f}, std={data.std():.4f}, n={len(data)}")
        
        # Resumen ELA (solo JPEG)
        print("\n--- ELA (solo JPEG) ---")
        df_jpeg = df[df['is_jpeg'] == True]
        for lbl in [0, 1]:
            data = df_jpeg[df_jpeg['label'] == lbl]['ela_mean']
            if len(data) > 0:
                print(f"  Label {lbl}: mean={data.mean():.4f}, std={data.std():.4f}, n={len(data)}")
            else:
                print(f"  Label {lbl}: sin datos JPEG")


if __name__ == "__main__":
    process_dataset()
