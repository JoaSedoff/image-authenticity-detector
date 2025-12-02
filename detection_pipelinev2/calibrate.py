"""
Script de Calibración para Pipeline Forense V2

Lee el dataset etiquetado, extrae métricas ELA y determina
umbrales óptimos mediante búsqueda exhaustiva.

Solo procesa imágenes JPEG para análisis ELA válido.
"""

import os
import sys
import csv
import numpy as np
from typing import Dict, List, Tuple, Optional

# Agregar path del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_pipelinev2.pipeline import ForensicPipeline
from detection_pipelinev2.utils import load_dataset_labels, is_jpeg


class ELACalibrator:
    """
    Calibrador para determinar umbrales óptimos de detección
    basados en métricas ELA y otras características.
    """
    
    def __init__(self, dataset_path: str, data_file: str = "data.txt"):
        """
        Args:
            dataset_path: Ruta a carpeta con imágenes
            data_file: Nombre del archivo de etiquetas
        """
        self.dataset_path = dataset_path
        self.data_file = os.path.join(dataset_path, data_file)
        self.pipeline = ForensicPipeline(ela_quality=90)
        self.features_data = []
    
    def extract_features(self, jpeg_only: bool = True, verbose: bool = True):
        """
        Extrae características de todas las imágenes etiquetadas.
        
        Args:
            jpeg_only: Solo procesar imágenes JPEG
            verbose: Mostrar progreso
        """
        # Cargar etiquetas
        labels = load_dataset_labels(self.data_file)
        
        if not labels:
            print(f"Error: No se encontraron etiquetas en {self.data_file}")
            return
        
        print(f"Etiquetas cargadas: {len(labels)} imágenes")
        
        # Filtrar solo JPEG si se requiere
        if jpeg_only:
            labels = {k: v for k, v in labels.items() if is_jpeg(k)}
            print(f"Imágenes JPEG: {len(labels)}")
        
        total = len(labels)
        processed = 0
        errors = 0
        
        for filename, label in labels.items():
            image_path = os.path.join(self.dataset_path, filename)
            
            if not os.path.exists(image_path):
                if verbose:
                    print(f"  [!] No encontrado: {filename}")
                errors += 1
                continue
            
            if verbose:
                print(f"  [{processed+1}/{total}] Procesando: {filename}...", end="\r")
            
            # Extraer características (solo ELA para velocidad)
            features = self.pipeline.extract_features_for_calibration(image_path, include_legacy=False)
            
            if features:
                features["filename"] = filename
                features["label"] = label  # 0=real, 1=IA
                self.features_data.append(features)
                processed += 1
            else:
                errors += 1
        
        print(f"\nExtracción completada: {processed} procesadas, {errors} errores")
    
    def find_optimal_thresholds(self) -> Dict[str, Dict]:
        """
        Encuentra umbrales óptimos para cada métrica.
        
        Usa búsqueda exhaustiva para maximizar accuracy.
        
        Returns:
            Diccionario con mejores umbrales por métrica
        """
        if not self.features_data:
            print("Error: No hay datos de características. Ejecutar extract_features() primero.")
            return {}
        
        # Obtener lista de métricas disponibles
        sample = self.features_data[0]
        metric_keys = [k for k in sample.keys() if k not in ["filename", "label"]]
        
        print(f"\nAnalizando {len(metric_keys)} métricas...")
        print("-" * 60)
        
        results = {}
        best_overall = {"metric": None, "accuracy": 0, "threshold": 0, "direction": None}
        
        for metric_key in metric_keys:
            # Extraer valores y etiquetas
            values = []
            labels = []
            
            for data in self.features_data:
                val = data.get(metric_key)
                if val is not None and not np.isnan(val):
                    values.append(val)
                    labels.append(data["label"])
            
            if len(values) < 5:
                continue
            
            values = np.array(values)
            labels = np.array(labels)
            
            # Buscar umbral óptimo
            best_thresh, best_acc, best_dir = self._search_threshold(values, labels)
            
            results[metric_key] = {
                "threshold": best_thresh,
                "accuracy": best_acc,
                "direction": best_dir,
                "mean_real": float(np.mean(values[labels == 0])) if np.sum(labels == 0) > 0 else 0,
                "mean_ia": float(np.mean(values[labels == 1])) if np.sum(labels == 1) > 0 else 0,
                "std_real": float(np.std(values[labels == 0])) if np.sum(labels == 0) > 0 else 0,
                "std_ia": float(np.std(values[labels == 1])) if np.sum(labels == 1) > 0 else 0
            }
            
            # Actualizar mejor global
            if best_acc > best_overall["accuracy"]:
                best_overall = {
                    "metric": metric_key,
                    "accuracy": best_acc,
                    "threshold": best_thresh,
                    "direction": best_dir
                }
            
            print(f"{metric_key:30s} | Acc: {best_acc*100:5.1f}% | Thresh: {best_thresh:8.4f} | {best_dir}")
        
        print("-" * 60)
        print(f"\n*** MEJOR MÉTRICA: {best_overall['metric']} ***")
        print(f"    Accuracy: {best_overall['accuracy']*100:.2f}%")
        print(f"    Umbral: {best_overall['threshold']:.4f}")
        print(f"    Dirección: {best_overall['direction']}")
        
        results["_best"] = best_overall
        
        return results
    
    def _search_threshold(self, values: np.ndarray, 
                           labels: np.ndarray) -> Tuple[float, float, str]:
        """
        Búsqueda exhaustiva del mejor umbral.
        
        Args:
            values: Array de valores de la métrica
            labels: Array de etiquetas (0=real, 1=IA)
            
        Returns:
            (mejor_umbral, mejor_accuracy, dirección)
        """
        # Generar candidatos de umbral
        min_val, max_val = np.min(values), np.max(values)
        thresholds = np.linspace(min_val, max_val, 200)
        
        best_acc = 0
        best_thresh = 0
        best_dir = "higher_is_fake"
        
        for thresh in thresholds:
            # Probar: higher_is_fake
            preds_high = (values > thresh).astype(int)
            acc_high = np.mean(preds_high == labels)
            
            if acc_high > best_acc:
                best_acc = acc_high
                best_thresh = thresh
                best_dir = "higher_is_fake"
            
            # Probar: lower_is_fake
            preds_low = (values < thresh).astype(int)
            acc_low = np.mean(preds_low == labels)
            
            if acc_low > best_acc:
                best_acc = acc_low
                best_thresh = thresh
                best_dir = "lower_is_fake"
        
        return float(best_thresh), float(best_acc), best_dir
    
    def save_features_csv(self, output_path: str = "dataset_features_v2.csv"):
        """
        Guarda características extraídas en CSV.
        
        Args:
            output_path: Ruta del archivo CSV de salida
        """
        if not self.features_data:
            print("Error: No hay datos para guardar.")
            return
        
        # Obtener todas las claves
        all_keys = set()
        for data in self.features_data:
            all_keys.update(data.keys())
        
        # Ordenar: filename, label primero, luego alfabético
        ordered_keys = ["filename", "label"]
        ordered_keys.extend(sorted([k for k in all_keys if k not in ordered_keys]))
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys)
            writer.writeheader()
            writer.writerows(self.features_data)
        
        print(f"Características guardadas en: {output_path}")
    
    def generate_report(self, results: Dict, output_path: str = "calibration_report.txt"):
        """
        Genera reporte de calibración en texto.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("REPORTE DE CALIBRACIÓN - PIPELINE FORENSE V2 (ELA)")
        lines.append("=" * 70)
        lines.append(f"\nTotal de imágenes analizadas: {len(self.features_data)}")
        
        n_real = sum(1 for d in self.features_data if d["label"] == 0)
        n_ia = sum(1 for d in self.features_data if d["label"] == 1)
        lines.append(f"  - Reales: {n_real}")
        lines.append(f"  - IA: {n_ia}")
        
        lines.append("\n" + "-" * 70)
        lines.append("RESULTADOS POR MÉTRICA")
        lines.append("-" * 70)
        
        # Ordenar por accuracy
        sorted_metrics = sorted(
            [(k, v) for k, v in results.items() if k != "_best"],
            key=lambda x: x[1]["accuracy"],
            reverse=True
        )
        
        for metric, data in sorted_metrics:
            lines.append(f"\n{metric}:")
            lines.append(f"  Accuracy: {data['accuracy']*100:.2f}%")
            lines.append(f"  Umbral: {data['threshold']:.4f}")
            lines.append(f"  Dirección: {data['direction']}")
            lines.append(f"  Media Real: {data['mean_real']:.4f} (±{data['std_real']:.4f})")
            lines.append(f"  Media IA: {data['mean_ia']:.4f} (±{data['std_ia']:.4f})")
        
        if "_best" in results:
            best = results["_best"]
            lines.append("\n" + "=" * 70)
            lines.append("CONFIGURACIÓN ÓPTIMA RECOMENDADA")
            lines.append("=" * 70)
            lines.append(f"Métrica: {best['metric']}")
            lines.append(f"Umbral: {best['threshold']:.4f}")
            lines.append(f"Dirección: {best['direction']}")
            lines.append(f"Accuracy esperada: {best['accuracy']*100:.2f}%")
        
        report_text = "\n".join(lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\nReporte guardado en: {output_path}")
        print(report_text)


def main():
    """Ejecutar calibración con dataset por defecto."""
    # Ruta al dataset
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Imagenes", "DeteccionIA"
    )
    
    if not os.path.exists(dataset_path):
        print(f"Error: No se encontró el dataset en {dataset_path}")
        return
    
    print("=" * 60)
    print("CALIBRACIÓN PIPELINE FORENSE V2 - ANÁLISIS ELA")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Modo: Solo imágenes JPEG\n")
    
    # Inicializar calibrador
    calibrator = ELACalibrator(dataset_path)
    
    # Extraer características
    print("Extrayendo características ELA...")
    calibrator.extract_features(jpeg_only=True, verbose=True)
    
    if not calibrator.features_data:
        print("No se extrajeron características. Verifica el dataset.")
        return
    
    # Guardar CSV
    calibrator.save_features_csv("dataset_features_ela.csv")
    
    # Buscar umbrales óptimos
    print("\nBuscando umbrales óptimos...")
    results = calibrator.find_optimal_thresholds()
    
    # Generar reporte
    calibrator.generate_report(results, "calibration_report_ela.txt")


if __name__ == "__main__":
    main()
