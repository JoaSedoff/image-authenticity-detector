# Scripts de Utilidad

Scripts auxiliares para etiquetado, extracción de características y análisis batch.

---

## labeling_tool.py

Herramienta gráfica (Tkinter) para etiquetar imágenes manualmente.

**Uso:**
```
python scripts/labeling_tool.py
```

**Funcionalidad:**
- Carga imágenes individuales mediante interfaz gráfica
- Muestra métricas calculadas (Laplaciano, FFT) antes de etiquetar
- Permite clasificar como REAL (0) o IA (1)
- Copia la imagen a `Imagenes/DeteccionIA/` con nombre secuencial
- Guarda etiqueta en `data.txt` y métricas en `dataset_metrics.csv`

**Archivos generados:**
- `Imagenes/DeteccionIA/imgX.ext` — Imagen copiada
- `Imagenes/DeteccionIA/data.txt` — Etiquetas (filename, label)
- `dataset_metrics.csv` — Métricas por imagen

---

## characteristic_extraction.py

Extrae características de todas las imágenes etiquetadas en el dataset.

**Uso:**
```
python scripts/characteristic_extraction.py
```

**Funcionalidad:**
- Lee `Imagenes/DeteccionIA/data.txt` para obtener imágenes etiquetadas
- Calcula métricas para cada imagen:
  - Histograma: media, std, entropía
  - Laplaciano: media, varianza
  - FFT: ratio alta/baja frecuencia, uniformidad radial, entropía espectral
- Exporta a CSV para análisis y calibración

**Archivo generado:**
- `dataset_features.csv` — Métricas completas del dataset

---

## deteccionv2.py

Clasificador basado en umbrales calibrados y análisis batch de directorios.

**Uso como módulo:**
```python
from scripts.deteccionv2 import classify_metrics

prediction, confidence = classify_metrics(metrics_dict)
```

**Uso standalone (análisis batch):**
```
python scripts/deteccionv2.py
```

**Funcionalidad:**
- `classify_metrics(metrics)`: Clasifica imagen según umbral configurado
- `AutoDetectorV2`: Analiza directorios completos recursivamente
- Genera reporte estadístico y sugiere umbrales óptimos

**Configuración actual:**
```python
CLASSIFICATION_CONFIG = {
    "primary_feature": "laplacian_score",
    "threshold": 4.5853,
    "direction": "higher_is_fake"
}
```

---

## Flujo de trabajo típico

1. **Etiquetar imágenes**: `python scripts/labeling_tool.py`
2. **Extraer características**: `python scripts/characteristic_extraction.py`
3. **Calibrar umbrales**: `python -m detection_pipelinev2.calibrate`
4. **Probar clasificación**: `python scripts/deteccionv2.py`
