"""
Script de Calibración para Pipeline Forense V2

Lee el CSV con características extraídas y determina
umbrales óptimos para:
- Laplaciano (todas las imágenes)
- ELA (solo imágenes JPEG)

Soporta modo automático (discriminación por formato).
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Agregar path del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CombinedCalibrator:
    """
    Calibrador que determina umbrales óptimos para múltiples algoritmos
    usando datos pre-extraídos de un CSV.
    """
    
    def __init__(self, csv_path: str = "dataset_features_combined.csv"):
        """
        Args:
            csv_path: Ruta al CSV con características extraídas
        """
        self.csv_path = csv_path
        self.df = None
        self.df_jpeg = None
        self.df_png = None
        
    def load_data(self):
        """Carga datos del CSV."""
        if not os.path.exists(self.csv_path):
            print(f"ERROR: No se encontró {self.csv_path}")
            return False
        
        self.df = pd.read_csv(self.csv_path)
        
        # Separar por formato
        self.df_jpeg = self.df[self.df['is_jpeg'] == True].copy()
        self.df_png = self.df[self.df['is_jpeg'] == False].copy()
        
        print(f"Datos cargados: {len(self.df)} imágenes")
        print(f"  - JPEG: {len(self.df_jpeg)} (Real: {len(self.df_jpeg[self.df_jpeg['label']==0])}, IA: {len(self.df_jpeg[self.df_jpeg['label']==1])})")
        print(f"  - PNG:  {len(self.df_png)} (Real: {len(self.df_png[self.df_png['label']==0])}, IA: {len(self.df_png[self.df_png['label']==1])})")
        
        return True
    
    def _search_threshold(self, values: np.ndarray, 
                          labels: np.ndarray) -> Tuple[float, float, str]:
        """
        Búsqueda exhaustiva del mejor umbral.
        """
        # Filtrar NaN
        mask = ~np.isnan(values)
        values = values[mask]
        labels = labels[mask]
        
        if len(values) < 5:
            return 0.0, 0.0, "higher_is_fake"
        
        min_val, max_val = np.min(values), np.max(values)
        thresholds = np.linspace(min_val, max_val, 200)
        
        best_acc = 0
        best_thresh = 0
        best_dir = "higher_is_fake"
        
        for thresh in thresholds:
            # higher_is_fake
            preds_high = (values > thresh).astype(int)
            acc_high = np.mean(preds_high == labels)
            
            if acc_high > best_acc:
                best_acc = acc_high
                best_thresh = thresh
                best_dir = "higher_is_fake"
            
            # lower_is_fake
            preds_low = (values < thresh).astype(int)
            acc_low = np.mean(preds_low == labels)
            
            if acc_low > best_acc:
                best_acc = acc_low
                best_thresh = thresh
                best_dir = "lower_is_fake"
        
        return float(best_thresh), float(best_acc), best_dir
    
    def calibrate_laplacian(self, data: pd.DataFrame = None, name: str = "todas") -> Dict:
        """
        Calibra umbral para Laplaciano.
        
        Args:
            data: DataFrame a usar (default: todas las imágenes)
            name: Nombre descriptivo del conjunto
        """
        if data is None:
            data = self.df
        
        values = data['laplacian_mean'].values
        labels = data['label'].values
        
        thresh, acc, direction = self._search_threshold(values, labels)
        
        result = {
            "metric": "laplacian_mean",
            "threshold": thresh,
            "accuracy": acc,
            "direction": direction,
            "n_samples": len(data),
            "mean_real": float(data[data['label']==0]['laplacian_mean'].mean()),
            "mean_ia": float(data[data['label']==1]['laplacian_mean'].mean()),
            "std_real": float(data[data['label']==0]['laplacian_mean'].std()),
            "std_ia": float(data[data['label']==1]['laplacian_mean'].std()),
        }
        
        print(f"\n--- LAPLACIANO ({name}) ---")
        print(f"  Samples: {result['n_samples']}")
        print(f"  Threshold: {result['threshold']:.4f}")
        print(f"  Accuracy: {result['accuracy']*100:.2f}%")
        print(f"  Direction: {result['direction']}")
        print(f"  Mean Real: {result['mean_real']:.4f} (±{result['std_real']:.4f})")
        print(f"  Mean IA: {result['mean_ia']:.4f} (±{result['std_ia']:.4f})")
        
        return result
    
    def calibrate_ela(self) -> Dict:
        """
        Calibra umbral para ELA (solo JPEG).
        """
        data = self.df_jpeg
        
        if len(data) < 5:
            print("\n--- ELA (JPEG) ---")
            print("  ERROR: No hay suficientes imágenes JPEG para calibrar")
            return {}
        
        values = data['ela_mean'].values
        labels = data['label'].values
        
        thresh, acc, direction = self._search_threshold(values, labels)
        
        result = {
            "metric": "ela_mean",
            "threshold": thresh,
            "accuracy": acc,
            "direction": direction,
            "n_samples": len(data),
            "mean_real": float(data[data['label']==0]['ela_mean'].mean()),
            "mean_ia": float(data[data['label']==1]['ela_mean'].mean()),
            "std_real": float(data[data['label']==0]['ela_mean'].std()),
            "std_ia": float(data[data['label']==1]['ela_mean'].std()),
        }
        
        print(f"\n--- ELA (solo JPEG) ---")
        print(f"  Samples: {result['n_samples']}")
        print(f"  Threshold: {result['threshold']:.4f}")
        print(f"  Accuracy: {result['accuracy']*100:.2f}%")
        print(f"  Direction: {result['direction']}")
        print(f"  Mean Real: {result['mean_real']:.4f} (±{result['std_real']:.4f})")
        print(f"  Mean IA: {result['mean_ia']:.4f} (±{result['std_ia']:.4f})")
        
        return result
    
    def calibrate_auto_mode(self) -> Dict:
        """
        Calcula accuracy del modo automático:
        - JPEG -> ELA
        - PNG -> Laplaciano
        """
        # Calibrar ELA para JPEG
        ela_result = self.calibrate_ela()
        
        # Calibrar Laplaciano para PNG
        lap_png_result = self.calibrate_laplacian(self.df_png, "solo PNG")
        
        # Calibrar Laplaciano para todas (referencia)
        lap_all_result = self.calibrate_laplacian(self.df, "todas")
        
        # Calcular accuracy combinada del modo AUTO
        correct = 0
        total = 0
        
        # JPEG con ELA
        if ela_result:
            for _, row in self.df_jpeg.iterrows():
                val = row['ela_mean']
                label = row['label']
                if pd.isna(val):
                    continue
                
                if ela_result['direction'] == 'lower_is_fake':
                    pred = 1 if val < ela_result['threshold'] else 0
                else:
                    pred = 1 if val > ela_result['threshold'] else 0
                
                if pred == label:
                    correct += 1
                total += 1
        
        # PNG con Laplaciano
        if lap_png_result:
            for _, row in self.df_png.iterrows():
                val = row['laplacian_mean']
                label = row['label']
                
                if lap_png_result['direction'] == 'higher_is_fake':
                    pred = 1 if val > lap_png_result['threshold'] else 0
                else:
                    pred = 1 if val < lap_png_result['threshold'] else 0
                
                if pred == label:
                    correct += 1
                total += 1
        
        auto_accuracy = correct / total if total > 0 else 0
        
        print(f"\n{'='*60}")
        print("MODO AUTOMÁTICO (discriminación por formato)")
        print(f"{'='*60}")
        print(f"  JPEG ({len(self.df_jpeg)} imgs) -> ELA: {ela_result.get('accuracy', 0)*100:.2f}%")
        print(f"  PNG  ({len(self.df_png)} imgs) -> Laplaciano: {lap_png_result.get('accuracy', 0)*100:.2f}%")
        print(f"\n  *** ACCURACY COMBINADA: {auto_accuracy*100:.2f}% ***")
        
        return {
            "ela": ela_result,
            "laplacian_png": lap_png_result,
            "laplacian_all": lap_all_result,
            "auto_accuracy": auto_accuracy,
            "total_samples": total
        }
    
    def generate_config_update(self, results: Dict) -> str:
        """
        Genera código Python para actualizar ALGORITHM_CONFIG en multi_detector.py
        """
        ela = results.get('ela', {})
        lap_png = results.get('laplacian_png', {})
        lap_all = results.get('laplacian_all', {})
        
        code = '''
# === CONFIGURACIÓN ACTUALIZADA (generada por calibrate.py) ===
# Fecha: {date}
# Accuracy modo AUTO: {auto_acc:.2f}%

ALGORITHM_CONFIG = {{
    "laplacian": {{
        "name": "Laplaciano",
        "description": "Análisis de alta frecuencia mediante filtro Laplaciano 3x3",
        "feature": "laplacian_score",
        "threshold": {lap_thresh:.4f},
        "direction": "{lap_dir}",
        "accuracy": {lap_acc:.2f},
        "unit": "Unidad 3: Filtros de vecindad",
        "best_for": ["png", "webp", "bmp", "tiff"]
    }},
    "ela": {{
        "name": "ELA (Error Level Analysis)",
        "description": "Análisis de niveles de error por recompresión JPEG",
        "feature": "ela_mean",
        "threshold": {ela_thresh:.4f},
        "direction": "{ela_dir}",
        "accuracy": {ela_acc:.2f},
        "unit": "Unidad 3: Operaciones aritméticas",
        "best_for": ["jpeg", "jpg"]
    }},
    ...
}}
'''.format(
            date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            auto_acc=results.get('auto_accuracy', 0) * 100,
            lap_thresh=lap_all.get('threshold', 4.5853),
            lap_dir=lap_all.get('direction', 'higher_is_fake'),
            lap_acc=lap_all.get('accuracy', 0) * 100,
            ela_thresh=ela.get('threshold', 0.1839),
            ela_dir=ela.get('direction', 'lower_is_fake'),
            ela_acc=ela.get('accuracy', 0) * 100,
        )
        
        return code
    
    def save_report(self, results: Dict, output_path: str = "calibration_report_combined.txt"):
        """Guarda reporte de calibración."""
        lines = []
        lines.append("=" * 70)
        lines.append("REPORTE DE CALIBRACIÓN COMBINADA - LAPLACIANO + ELA")
        lines.append("=" * 70)
        lines.append(f"\nFecha: {pd.Timestamp.now()}")
        lines.append(f"Dataset: {self.csv_path}")
        lines.append(f"Total imágenes: {len(self.df)}")
        lines.append(f"  - JPEG: {len(self.df_jpeg)}")
        lines.append(f"  - PNG: {len(self.df_png)}")
        
        lines.append("\n" + "-" * 70)
        lines.append("RESULTADOS POR ALGORITMO")
        lines.append("-" * 70)
        
        # ELA
        ela = results.get('ela', {})
        if ela:
            lines.append(f"\nELA (solo JPEG - {ela.get('n_samples', 0)} imgs):")
            lines.append(f"  Accuracy: {ela.get('accuracy', 0)*100:.2f}%")
            lines.append(f"  Threshold: {ela.get('threshold', 0):.4f}")
            lines.append(f"  Direction: {ela.get('direction', '')}")
            lines.append(f"  Mean Real: {ela.get('mean_real', 0):.4f}")
            lines.append(f"  Mean IA: {ela.get('mean_ia', 0):.4f}")
        
        # Laplaciano PNG
        lap_png = results.get('laplacian_png', {})
        if lap_png:
            lines.append(f"\nLAPLACIANO (solo PNG - {lap_png.get('n_samples', 0)} imgs):")
            lines.append(f"  Accuracy: {lap_png.get('accuracy', 0)*100:.2f}%")
            lines.append(f"  Threshold: {lap_png.get('threshold', 0):.4f}")
            lines.append(f"  Direction: {lap_png.get('direction', '')}")
        
        # Laplaciano todas
        lap_all = results.get('laplacian_all', {})
        if lap_all:
            lines.append(f"\nLAPLACIANO (todas - {lap_all.get('n_samples', 0)} imgs):")
            lines.append(f"  Accuracy: {lap_all.get('accuracy', 0)*100:.2f}%")
            lines.append(f"  Threshold: {lap_all.get('threshold', 0):.4f}")
            lines.append(f"  Direction: {lap_all.get('direction', '')}")
        
        lines.append("\n" + "=" * 70)
        lines.append("MODO AUTOMÁTICO (RECOMENDADO)")
        lines.append("=" * 70)
        lines.append(f"JPEG -> ELA | PNG -> Laplaciano")
        lines.append(f"*** ACCURACY COMBINADA: {results.get('auto_accuracy', 0)*100:.2f}% ***")
        
        lines.append("\n" + "-" * 70)
        lines.append("CÓDIGO PARA ACTUALIZAR multi_detector.py:")
        lines.append("-" * 70)
        lines.append(self.generate_config_update(results))
        
        report = "\n".join(lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nReporte guardado en: {output_path}")
        return report


def main():
    """Ejecutar calibración combinada."""
    print("=" * 60)
    print("CALIBRACIÓN COMBINADA - LAPLACIANO + ELA")
    print("=" * 60)
    
    # Buscar CSV en directorio actual o raíz del proyecto
    csv_paths = [
        "dataset_features_combined.csv",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset_features_combined.csv")
    ]
    
    csv_path = None
    for path in csv_paths:
        if os.path.exists(path):
            csv_path = path
            break
    
    if not csv_path:
        print("ERROR: No se encontró dataset_features_combined.csv")
        print("Ejecuta primero: python scripts/characteristic_extraction.py")
        return
    
    print(f"CSV: {csv_path}\n")
    
    # Calibrar
    calibrator = CombinedCalibrator(csv_path)
    
    if not calibrator.load_data():
        return
    
    results = calibrator.calibrate_auto_mode()
    
    # Guardar reporte
    report = calibrator.save_report(results)
    print("\n" + report)


if __name__ == "__main__":
    main()
