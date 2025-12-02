# Detection Pipeline V2
# Técnicas forenses clásicas de PDI para detección de imágenes IA
# Sin machine learning - basado en programa ICO 527

from .pipeline import ForensicPipeline
from .classifier import classify_with_ela, classify_combined, update_ela_config

__version__ = "2.0.0"
__all__ = ["ForensicPipeline", "classify_with_ela", "classify_combined", "update_ela_config"]
