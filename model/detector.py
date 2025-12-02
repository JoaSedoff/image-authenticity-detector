"""
Módulo de detección de imágenes generadas por IA
Utiliza técnicas de PDI clásico: histogramas, Laplaciano y FFT
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para servidor
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import sys
import os

# Intentar importar la lógica de clasificación desde deteccionv2
try:
    # Si estamos corriendo desde root
    import deteccionv2
except ImportError:
    # Si estamos corriendo desde model/ o similar, intentar ajustar path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        import deteccionv2
    except ImportError:
        print("Advertencia: No se pudo importar deteccionv2. Usando lógica fallback.")
        deteccionv2 = None


class AIImageDetector:
    """Detector de imágenes generadas por IA usando técnicas de PDI clásico"""
    
    def __init__(self):
        # Definición del kernel Laplaciano 3x3
        self.kernel_lap = np.array([[0, -1, 0],
                                     [-1, 4, -1],
                                     [0, -1, 0]])
    
    def analyze_image(self, image_path, generate_plots=True):
        """
        Analiza una imagen y retorna métricas de detección
        
        Args:
            image_path: Ruta a la imagen a analizar
            generate_plots: Si es True, genera gráficos en base64 (lento). Si es False, solo métricas.
            
        Returns:
            dict: Diccionario con resultados del análisis
        """
        # Cargar imagen soportando caracteres especiales en la ruta (Windows)
        try:
            # Intentar carga normal primero
            img = cv2.imread(image_path)
            
            # Si falla, intentar con numpy (para rutas con acentos en Windows)
            if img is None:
                with open(image_path, "rb") as f:
                    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            return {"error": f"Excepción al cargar imagen: {str(e)}"}

        if img is None:
            return {"error": "No se pudo cargar la imagen"}
        
        # Redimensionar si es muy grande para optimizar memoria
        max_dimension = 4000
        height, width = img.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"Imagen redimensionada de {width}x{height} a {new_width}x{new_height}")
        
        # Convertir de BGR a RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convertir a escala de grises
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # 1. Análisis de histograma
        histogram = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
        hist_plot = self._plot_histogram(histogram, "Histograma de la imagen") if generate_plots else None
        
        # 2. Análisis Laplaciano (detección de bordes)
        laplacian = cv2.filter2D(img_gray, -1, self.kernel_lap)
        laplacian_score = np.mean(np.abs(laplacian))
        lap_plot = self._plot_laplacian(laplacian, "Mapa Laplaciano") if generate_plots else None
        
        # 3. Análisis FFT (espectro de frecuencias)
        fft_metrics, fft_plot = self._analyze_fft(img_gray, generate_plot=generate_plots)
        fft_score = fft_metrics['high_freq_energy']
        
        # 4. Clasificación
        # Si deteccionv2 está disponible, usar su lógica centralizada
        if deteccionv2:
            # Pasar todas las métricas disponibles, incluyendo laplacian_score
            all_metrics = fft_metrics.copy()
            all_metrics['laplacian_score'] = laplacian_score
            prediction, confidence = deteccionv2.classify_metrics(all_metrics)
        else:
            # Fallback a lógica antigua
            prediction, confidence = self._classify_image(laplacian_score, fft_score)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "laplacian_score": round(laplacian_score, 2),
            "fft_score": round(fft_score, 2),
            "fft_metrics": fft_metrics,
            "histogram_plot": hist_plot,
            "laplacian_plot": lap_plot,
            "fft_plot": fft_plot,
            "dimensions": img_rgb.shape
        }
    
    def _plot_histogram(self, histogram, title):
        """Genera gráfico del histograma y lo retorna como base64"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(histogram)
        ax.set_title(title)
        ax.set_xlabel('Valor de Píxel')
        ax.set_ylabel('Frecuencia')
        plt.tight_layout()
        
        # Convertir a base64 con compresión optimizada
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return f"data:image/png;base64,{image_base64}"
    
    def _plot_laplacian(self, laplacian, title):
        """Genera visualización del mapa Laplaciano"""
        # Redimensionar si es muy grande para la visualización
        if max(laplacian.shape) > 800:
            scale = 800 / max(laplacian.shape)
            new_size = (int(laplacian.shape[1] * scale), int(laplacian.shape[0] * scale))
            laplacian = cv2.resize(laplacian, new_size, interpolation=cv2.INTER_AREA)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(laplacian, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return f"data:image/png;base64,{image_base64}"
    
    def _analyze_fft(self, img_gray, generate_plot=True):
        """Análisis espectral usando FFT 2D"""
        # Reducir tamaño para optimizar memoria y cálculo FFT
        # Para imágenes muy grandes, reducir más agresivamente
        max_size = max(img_gray.shape)
        if max_size > 2000:
            scale_factor = 0.05  # Reducir más para imágenes muy grandes
        else:
            scale_factor = 0.1
        
        img_small = cv2.resize(img_gray, None, fx=scale_factor, fy=scale_factor, 
                              interpolation=cv2.INTER_AREA)
        
        # FFT 2D
        f = np.fft.fft2(img_small)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # Calcular métrica de ruido espectral (High Frequency Energy)
        center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
        mask_size = min(center_y, center_x) // 4
        
        # Crear máscara para altas frecuencias (excluyendo el centro)
        mask = np.ones(magnitude_spectrum.shape, dtype=bool)
        mask[center_y - mask_size:center_y + mask_size, 
             center_x - mask_size:center_x + mask_size] = False
        
        high_freq_energy = np.mean(magnitude_spectrum[mask])

        # --- Métricas Avanzadas (del Notebook) ---
        magnitud = np.abs(fshift)
        h, w = magnitud.shape
        cy, cx = h // 2, w // 2
        
        # 1. Ratio Alta/Baja Frecuencia
        radio_bajo = min(h, w) * 0.15
        y, x = np.ogrid[:h, :w]
        dist_sq = (x - cx)**2 + (y - cy)**2
        mascara_bajo = dist_sq <= radio_bajo**2
        
        energia_total = np.sum(magnitud**2)
        energia_baja = np.sum(magnitud[mascara_bajo]**2)
        energia_alta = energia_total - energia_baja
        ratio_af_bf = energia_alta / energia_baja if energia_baja > 0 else 0

        # 2. Uniformidad Radial
        max_radio = min(h, w) // 2
        num_anillos = 20
        energias_anillos = []
        dist = np.sqrt(dist_sq)
        
        for i in range(num_anillos):
            r_inner = (i / num_anillos) * max_radio
            r_outer = ((i + 1) / num_anillos) * max_radio
            mascara_anillo = (dist >= r_inner) & (dist < r_outer)
            if np.sum(mascara_anillo) > 0:
                energia_anillo = np.mean(magnitud[mascara_anillo])
                energias_anillos.append(energia_anillo)
        
        uniformidad_radial = np.std(energias_anillos) / (np.mean(energias_anillos) + 1e-10)

        # 3. Entropía Espectral
        magnitud_norm = magnitud / (np.sum(magnitud) + 1e-10)
        magnitud_norm = magnitud_norm[magnitud_norm > 0]
        entropia = -np.sum(magnitud_norm * np.log(magnitud_norm + 1e-10))

        # 4. Ratio Frecuencias Medias (15%-40%)
        r_min = max_radio * 0.15
        r_max = max_radio * 0.40
        mascara_media = (dist >= r_min) & (dist < r_max)
        energia_media = np.sum(magnitud[mascara_media]**2)
        ratio_freq_media = energia_media / energia_total if energia_total > 0 else 0

        metrics = {
            'high_freq_energy': high_freq_energy,
            'ratio_af_bf': ratio_af_bf,
            'uniformidad_radial': uniformidad_radial,
            'entropia': entropia,
            'ratio_freq_media': ratio_freq_media
        }
        
        fft_plot = None
        if generate_plot:
            # Generar visualización
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(magnitude_spectrum, cmap='gray')
            ax.set_title('Espectro de Frecuencias (FFT 2D)')
            ax.axis('off')
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            fft_plot = f"data:image/png;base64,{image_base64}"
        
        return metrics, fft_plot
    
    def _classify_image(self, laplacian_score, fft_score):
        """
        Clasifica la imagen basándose en umbrales empíricos
        
        Según el análisis del proyecto:
        - Imagen real: Laplaciano ~1.22, bajo ruido espectral
        - Imágenes IA: Laplaciano 4-7, mayor ruido espectral
        """
        # Umbrales basados en los datos del proyecto
        LAPLACIAN_THRESHOLD = 2.5
        
        if laplacian_score < LAPLACIAN_THRESHOLD:
            prediction = "REAL"
            confidence = min(95, 70 + (LAPLACIAN_THRESHOLD - laplacian_score) * 10)
        else:
            prediction = "GENERADA POR IA"
            confidence = min(95, 60 + (laplacian_score - LAPLACIAN_THRESHOLD) * 5)
        
        return prediction, round(confidence, 1)


def compare_images(image_paths):
    """
    Compara múltiples imágenes y genera un análisis comparativo
    
    Args:
        image_paths: Lista de rutas a las imágenes
        
    Returns:
        dict: Resultados comparativos
    """
    detector = AIImageDetector()
    results = []
    
    for path in image_paths:
        result = detector.analyze_image(path)
        results.append(result)
    
    return results
