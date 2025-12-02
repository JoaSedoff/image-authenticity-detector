import cv2
import numpy as np
import pandas as pd
import os
import sys

# Agregar directorio raíz del proyecto al path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # Cambiar al directorio raíz para rutas relativas

# Configuración de rutas
BASE_DIR = os.path.join('Imagenes', 'DetecciónIA')
DATA_FILE = os.path.join(BASE_DIR, 'data.txt')
OUTPUT_FILE = 'dataset_features.csv'

def get_histogram_metrics(gray_img):
    """
    Calcula métricas estadísticas básicas basadas en el histograma y la intensidad.
    """
    # Estadísticas directas de la imagen
    mean_val = np.mean(gray_img)
    std_val = np.std(gray_img)
    
    # Calcular histograma para entropía de Shannon (opcional, pero útil)
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    hist_norm = hist_norm[hist_norm > 0]
    entropy = -np.sum(hist_norm * np.log2(hist_norm))
    
    return {
        'intensity_mean': mean_val,
        'intensity_std': std_val,
        'intensity_entropy': entropy
    }

def get_laplacian_metric(gray_img):
    """
    Calcula métrica de alta frecuencia usando filtro Laplaciano.
    Basado en la sección 3 del notebook Detección.ipynb.
    """
    kernel_lap = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])
    
    lap_img = cv2.filter2D(gray_img, -1, kernel_lap)
    
    # Métrica: Promedio de magnitudes absolutas (indica cantidad de bordes/ruido HF)
    hf_metric = np.mean(np.abs(lap_img))
    
    # Varianza del Laplaciano (otra medida común de nitidez/ruido)
    lap_var = np.var(lap_img)
    
    return {
        'laplacian_mean': hf_metric,
        'laplacian_var': lap_var
    }

def get_spectral_metrics(gray_img):
    """
    Calcula métricas en el dominio de la frecuencia (FFT).
    Basado en las secciones 4 y 5 del notebook Detección.ipynb.
    """
    # Redimensionar para consistencia y velocidad (como en el notebook)
    # Usamos un tamaño fijo razonable (e.g., 512px) para normalizar el análisis espectral
    h, w = gray_img.shape
    target_size = 512
    scale = min(1.0, target_size / max(h, w))
    
    if scale < 1.0:
        img_small = cv2.resize(gray_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img_small = gray_img

    # FFT 2D
    f = np.fft.fft2(img_small)
    fshift = np.fft.fftshift(f)
    magnitud = np.abs(fshift)
    
    h_fft, w_fft = magnitud.shape
    cy, cx = h_fft // 2, w_fft // 2
    y, x = np.ogrid[:h_fft, :w_fft]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_radio = min(h_fft, w_fft) // 2
    
    # 1. Ratio Alta/Baja Frecuencia (Definición del notebook: baja < 15% radio)
    radio_bajo = min(h_fft, w_fft) * 0.15
    mascara_bajo = dist <= radio_bajo
    
    energia_total = np.sum(magnitud**2)
    energia_baja = np.sum(magnitud[mascara_bajo]**2)
    energia_alta = energia_total - energia_baja
    
    ratio_af_bf = energia_alta / energia_baja if energia_baja > 0 else 0
    
    # 2. Uniformidad Radial (Variación de energía en anillos concéntricos)
    num_anillos = 20
    energias_anillos = []
    for i in range(num_anillos):
        r_inner = (i / num_anillos) * max_radio
        r_outer = ((i + 1) / num_anillos) * max_radio
        mascara_anillo = (dist >= r_inner) & (dist < r_outer)
        
        if np.sum(mascara_anillo) > 0:
            energia_anillo = np.mean(magnitud[mascara_anillo])
            energias_anillos.append(energia_anillo)
            
    uniformidad_radial = np.std(energias_anillos) / (np.mean(energias_anillos) + 1e-10)
    
    # 3. Entropía Espectral (Regularidad de la distribución de energía)
    magnitud_norm = magnitud / (np.sum(magnitud) + 1e-10)
    magnitud_norm = magnitud_norm[magnitud_norm > 0]
    fft_entropia = -np.sum(magnitud_norm * np.log(magnitud_norm + 1e-10))
    
    # 4. Ratio Frecuencias Medias (Banda 15%-40%)
    r_min = max_radio * 0.15
    r_max = max_radio * 0.40
    mascara_media = (dist >= r_min) & (dist < r_max)
    energia_media = np.sum(magnitud[mascara_media]**2)
    ratio_media = energia_media / energia_total if energia_total > 0 else 0
    
    return {
        'fft_ratio_af_bf': ratio_af_bf,
        'fft_uniformidad': uniformidad_radial,
        'fft_entropia': fft_entropia,
        'fft_ratio_media': ratio_media
    }

def process_dataset():
    print(f"Iniciando extracción de características...")
    print(f"Leyendo etiquetas de: {DATA_FILE}")
    
    results = []
    
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: No se encontró el archivo de datos: {DATA_FILE}")
        return

    try:
        with open(DATA_FILE, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"ERROR al leer el archivo: {e}")
        return

    count = 0
    for line in lines:
        line = line.strip()
        if not line: continue
        
        parts = line.split(',')
        if len(parts) < 2: continue
        
        filename = parts[0].strip()
        try:
            label = int(parts[1].strip())
        except ValueError:
            print(f"Saltando línea inválida: {line}")
            continue
        
        img_path = os.path.join(BASE_DIR, filename)
        
        if not os.path.exists(img_path):
            print(f"Advertencia: Imagen no encontrada {img_path}")
            continue
            
        # Leer imagen usando método seguro para caracteres especiales en Windows
        try:
            # cv2.imread falla con caracteres especiales en Windows, usamos imdecode
            stream = open(img_path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
            stream.close()
        except Exception as e:
            print(f"Excepción al leer {filename}: {e}")
            img = None

        if img is None:
            print(f"Error al leer imagen: {filename}")
            continue
            
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extraer características
        features = {'filename': filename, 'label': label}
        
        # 1. Métricas de Histograma/Intensidad
        features.update(get_histogram_metrics(gray))
        
        # 2. Métricas Laplacianas (Bordes/HF)
        features.update(get_laplacian_metric(gray))
        
        # 3. Métricas Espectrales (FFT)
        features.update(get_spectral_metrics(gray))
        
        results.append(features)
        count += 1
        if count % 5 == 0:
            print(f"Procesadas {count} imágenes...")

    # Guardar resultados
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n=== PROCESAMIENTO COMPLETADO ===")
        print(f"Total imágenes procesadas: {len(df)}")
        print(f"Resultados guardados en: {OUTPUT_FILE}")
        
        print("\n=== RESUMEN DE PROMEDIOS POR CLASE (0=Real, 1=IA) ===")
        # Agrupar por etiqueta y mostrar promedios para ver diferencias
        summary = df.drop(columns=['filename']).groupby('label').mean()
        print(summary)
        
        print("\n=== ANÁLISIS PRELIMINAR ===")
        print("Compara los valores promedio entre la clase 0 y 1 para identificar umbrales.")
        print("Por ejemplo, si 'laplacian_mean' es mucho mayor en 1 que en 0, es un buen discriminador.")
    else:
        print("No se procesaron imágenes. Verifica las rutas y el archivo data.txt")

if __name__ == "__main__":
    process_dataset()
