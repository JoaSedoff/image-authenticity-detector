# Detector de Imágenes Generadas por IA

## Descripción

Aplicación web para detección de imágenes generadas o editadas por inteligencia artificial mediante técnicas de **Procesamiento Digital de Imágenes (PDI) clásico**, sin uso de machine learning ni redes neuronales.

**Materia**: Fundamentos del Procesamiento Digital de Imágenes – ICO 527

---

## Algoritmos de Detección

El sistema implementa tres algoritmos de detección seleccionables desde la interfaz:

### 1. ELA (Error Level Analysis) — Accuracy: 84.38%

Analiza inconsistencias en niveles de compresión JPEG mediante recompresión y cálculo de diferencia absoluta.

- **Fundamento teórico**: Unidad 3 (Operaciones aritméticas), Unidad 4 (Extracción de regiones)
- **Principio**: Las imágenes reales presentan mayor error por historial de compresión; las sintéticas muestran error uniformemente bajo
- **Umbral calibrado**: `ela_mean < 0.1839` → Imagen IA
- **Métricas**: Media, varianza, entropía y uniformidad del mapa de error

### 2. Laplaciano — Accuracy: 68.52%

Detecta contenido de alta frecuencia mediante filtro Laplaciano 3×3.

- **Fundamento teórico**: Unidad 3 (Filtros de vecindad, detección de bordes)
- **Kernel**: `[[0,-1,0],[-1,4,-1],[0,-1,0]]`
- **Principio**: Las imágenes generadas por IA suelen presentar mayor contenido de alta frecuencia debido a artefactos de síntesis
- **Umbral calibrado**: `laplacian_score > 4.5853` → Imagen IA

### 3. Combinado (Votación Ponderada)

Combina los clasificadores ELA y Laplaciano mediante votación ponderada.

- **Pesos**: ELA (1.5) + Laplaciano (1.0)
- **Decisión**: `voto_ponderado > 0.5` → Imagen IA
- **Ventaja**: Mayor robustez ante casos límite

---

## Técnicas de PDI Utilizadas

| Técnica | Aplicación |
|---------|------------|
| Análisis de histogramas | Distribución de intensidades |
| Operaciones aritméticas | Diferencia absoluta (ELA) |
| Filtro Laplaciano | Detección de bordes y alta frecuencia |
| FFT 2D | Análisis espectral |
| Extracción de regiones | Análisis por bloques 8×8 |

---

## Requisitos

- Python 3.8+
- Dependencias: Flask, OpenCV, NumPy, Matplotlib

```
pip install -r requirements.txt
```

---

## Ejecución

```
python run.py
```

El servidor inicia en `http://localhost:5000` y muestra una IP de red para acceso desde otros dispositivos en la misma LAN.

---

## Uso

1. Acceder a la interfaz web
2. Seleccionar algoritmo de detección (ELA recomendado)
3. Cargar imagen (formatos soportados: JPG, JPEG, PNG, BMP, TIFF, WEBP)
4. El sistema retorna:
   - Clasificación: REAL / GENERADA POR IA
   - Nivel de confianza
   - Métricas numéricas (scores)
   - Visualizaciones: histograma, mapa Laplaciano, espectro FFT, mapa ELA

---

## Estructura del Proyecto

```
PDI Proyecto/
├── app/
│   ├── routes.py                 # Endpoints de la API
│   └── templates/index.html      # Interfaz web
├── model/
│   ├── detector.py               # Detector legacy
│   └── multi_detector.py         # Detector multi-algoritmo
├── detection_pipelinev2/
│   ├── analyzers/ela.py          # Analizador ELA
│   ├── pipeline.py               # Orquestador
│   ├── classifier.py             # Lógica de clasificación
│   └── calibrate.py              # Script de calibración
├── Imagenes/DeteccionIA/         # Dataset de calibración
├── run.py                        # Punto de entrada
└── requirements.txt
```

---

## API

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Interfaz web |
| `/analyze` | POST | Análisis de imagen. Parámetros: `image` (archivo), `algorithm` (ela/laplacian/combined) |
| `/algorithms` | GET | Lista de algoritmos disponibles |

---

## Calibración

El sistema fue calibrado con un dataset de 54 imágenes etiquetadas (32 JPEG procesadas para ELA).

Para recalibrar con nuevas imágenes:
```
python -m detection_pipelinev2.calibrate
```

Archivos generados:
- `dataset_features_ela.csv`: Métricas extraídas
- `calibration_report_ela.txt`: Umbrales óptimos por métrica

---

## Limitaciones

- ELA requiere imágenes JPEG para máxima efectividad
- Dataset de calibración limitado
- Diseñado para uso académico y demostración
- Solo funciona en red local (LAN)

---

## Tecnologías

- **Backend**: Python, Flask
- **Procesamiento**: OpenCV, NumPy
- **Visualización**: Matplotlib
- **Frontend**: HTML5, CSS3, JavaScript

---

## Referencias

Técnicas basadas en el programa de la materia ICO 527:
- Unidad 2: Elementos de la imagen y transformaciones
- Unidad 3: Filtros de imágenes
- Unidad 4: Análisis espectral y de regiones

---

**Proyecto desarrollado para ICO 527 — Fundamentos del Procesamiento Digital de Imágenes**
