import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import csv
import os
import shutil
import sys

# Agregar directorio raíz del proyecto al path para imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.detector import AIImageDetector

class LabelingTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Herramienta de Etiquetado de Imágenes - IA vs Real")
        self.detector = AIImageDetector()
        self.current_image_path = None
        self.current_metrics = None
        
        # Configuración del Dataset
        # Se asume que la carpeta Imágenes/DetecciónIA ya existe
        self.dataset_dir = os.path.join(PROJECT_ROOT, "Imagenes", "DetecciónIA")
        self.txt_file = os.path.join(self.dataset_dir, "data.txt")
        self.csv_file = os.path.join(PROJECT_ROOT, "dataset_metrics.csv") # Archivo auxiliar para guardar métricas calculadas

        # Crear directorio si no existe (aunque el usuario dijo que ya existe)
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        # Inicializar CSV de métricas si no existe
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "filename", 
                    "laplacian_score", 
                    "fft_high_freq_energy",
                    "ratio_af_bf",
                    "uniformidad_radial",
                    "entropia",
                    "ratio_freq_media",
                    "label"
                ])

        # Elementos de la UI
        self.frame_top = tk.Frame(root)
        self.frame_top.pack(pady=10)

        self.btn_load = tk.Button(self.frame_top, text="Cargar Imagen", command=self.load_image)
        self.btn_load.pack()

        self.lbl_image = tk.Label(root)
        self.lbl_image.pack(pady=10)

        self.lbl_metrics = tk.Label(root, text="Métricas: -", justify=tk.LEFT)
        self.lbl_metrics.pack(pady=10)

        self.frame_buttons = tk.Frame(root)
        self.frame_buttons.pack(pady=10)

        # Botones actualizados para 0 (Real) y 1 (IA)
        self.btn_real = tk.Button(self.frame_buttons, text="ES REAL (0)", command=lambda: self.save_label(0), bg="green", fg="white", state=tk.DISABLED, width=15, height=2)
        self.btn_real.pack(side=tk.LEFT, padx=20)

        self.btn_ai = tk.Button(self.frame_buttons, text="ES IA (1)", command=lambda: self.save_label(1), bg="red", fg="white", state=tk.DISABLED, width=15, height=2)
        self.btn_ai.pack(side=tk.RIGHT, padx=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not file_path:
            return

        self.current_image_path = file_path
        
        # Mostrar Imagen
        try:
            img = Image.open(file_path)
            img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(img)
            self.lbl_image.config(image=photo)
            self.lbl_image.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar imagen para visualización: {e}")
            return

        # Calcular Métricas
        try:
            results = self.detector.analyze_image(file_path)
            
            if "error" in results:
                messagebox.showerror("Error", results["error"])
                return

            fft_metrics = results["fft_metrics"]
            
            self.current_metrics = {
                "laplacian_score": results["laplacian_score"],
                "fft_high_freq_energy": fft_metrics["high_freq_energy"],
                "ratio_af_bf": fft_metrics["ratio_af_bf"],
                "uniformidad_radial": fft_metrics["uniformidad_radial"],
                "entropia": fft_metrics["entropia"],
                "ratio_freq_media": fft_metrics["ratio_freq_media"]
            }
            
            metrics_text = (
                f"Laplacian Score: {results['laplacian_score']:.2f}\n"
                f"FFT High Freq Energy: {fft_metrics['high_freq_energy']:.2f}\n"
                f"Ratio AF/BF: {fft_metrics['ratio_af_bf']:.4f}\n"
                f"Uniformidad Radial: {fft_metrics['uniformidad_radial']:.4f}\n"
                f"Entropía: {fft_metrics['entropia']:.4f}\n"
                f"Ratio Freq Media: {fft_metrics['ratio_freq_media']:.4f}"
            )
            
            self.lbl_metrics.config(text=metrics_text)
            
            self.btn_real.config(state=tk.NORMAL)
            self.btn_ai.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al analizar la imagen: {str(e)}")

    def get_next_index(self):
        """Determina el siguiente índice n para imgn.ext basado en data.txt"""
        if not os.path.exists(self.txt_file):
            return 0
        try:
            with open(self.txt_file, 'r') as f:
                lines = f.readlines()
                if not lines:
                    return 0
                
                max_idx = -1
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 1:
                        fname = parts[0].strip()
                        # fname es tipo img12.png
                        name_part = os.path.splitext(fname)[0]
                        if name_part.startswith('img'):
                            try:
                                # Extraer número después de 'img'
                                idx_str = name_part[3:]
                                if idx_str.isdigit():
                                    idx = int(idx_str)
                                    if idx > max_idx:
                                        max_idx = idx
                            except ValueError:
                                pass
                return max_idx + 1
        except Exception:
            return 0

    def save_label(self, label_value):
        if not self.current_metrics or not self.current_image_path:
            return

        try:
            # 1. Determinar nuevo nombre de archivo
            idx = self.get_next_index()
            ext = os.path.splitext(self.current_image_path)[1]
            new_filename = f"img{idx}{ext}"
            new_filepath = os.path.join(self.dataset_dir, new_filename)

            # 2. Copiar imagen al directorio del dataset
            shutil.copy2(self.current_image_path, new_filepath)

            # 3. Agregar entrada a data.txt
            with open(self.txt_file, 'a') as f:
                f.write(f"{new_filename}, {label_value}\n")

            # 4. Guardar métricas en CSV (usando el nuevo nombre para referencia)
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    new_filename, 
                    self.current_metrics["laplacian_score"], 
                    self.current_metrics["fft_high_freq_energy"],
                    self.current_metrics["ratio_af_bf"],
                    self.current_metrics["uniformidad_radial"],
                    self.current_metrics["entropia"],
                    self.current_metrics["ratio_freq_media"],
                    label_value
                ])
            
            label_str = "REAL" if label_value == 0 else "IA"
            messagebox.showinfo("Guardado", f"Imagen guardada como {new_filename} ({label_str})")
            
            # Resetear UI
            self.btn_real.config(state=tk.DISABLED)
            self.btn_ai.config(state=tk.DISABLED)
            self.lbl_image.config(image='')
            self.lbl_metrics.config(text="Métricas: -")
            self.current_image_path = None
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LabelingTool(root)
    root.mainloop()
