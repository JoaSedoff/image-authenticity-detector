"""
Clasificador Forense V2

Lógica de clasificación basada en métricas ELA y combinadas.
Extensión del clasificador existente en deteccionv2.py.
"""

import numpy as np
from typing import Dict, Tuple, Optional


# Configuración de umbrales ELA (valores por defecto, se actualizan con calibración)
ELA_CONFIG = {
    "primary_feature": "ela_mean",
    "threshold": 5.0,
    "direction": "higher_is_fake",
    "enabled": True
}


def classify_with_ela(metrics: Dict[str, float]) -> Tuple[str, float, Dict]:
    """
    Clasifica imagen usando métricas ELA.
    
    Args:
        metrics: Diccionario con métricas extraídas
        
    Returns:
        (predicción, confianza, detalles)
    """
    if not ELA_CONFIG["enabled"]:
        return "DESCONOCIDO", 0.0, {"reason": "ELA deshabilitado"}
    
    feature = ELA_CONFIG["primary_feature"]
    threshold = ELA_CONFIG["threshold"]
    direction = ELA_CONFIG["direction"]
    
    val = metrics.get(feature)
    
    if val is None:
        return "DESCONOCIDO", 0.0, {"reason": f"Métrica {feature} no disponible"}
    
    is_fake = False
    if direction == "higher_is_fake":
        is_fake = val > threshold
    else:
        is_fake = val < threshold
    
    # Calcular confianza basada en distancia al umbral
    distance = abs(val - threshold)
    max_expected_distance = threshold * 0.5  # Heurística
    confidence = min(95.0, 60.0 + (distance / max_expected_distance) * 35.0)
    
    prediction = "GENERADA POR IA" if is_fake else "REAL"
    
    details = {
        "feature_used": feature,
        "value": val,
        "threshold": threshold,
        "direction": direction,
        "distance": distance
    }
    
    return prediction, round(confidence, 2), details


def classify_combined(metrics: Dict[str, float], 
                      weights: Optional[Dict[str, float]] = None) -> Tuple[str, float, Dict]:
    """
    Clasificación combinada usando múltiples métricas.
    
    Combina ELA con Laplaciano y FFT mediante votación ponderada.
    
    Args:
        metrics: Diccionario con todas las métricas
        weights: Pesos para cada clasificador (default: igual peso)
        
    Returns:
        (predicción, confianza, detalles)
    """
    if weights is None:
        weights = {
            "ela": 1.0,
            "laplacian": 1.0,
            "fft": 0.5
        }
    
    votes = []
    details = {}
    
    # Voto ELA
    ela_pred, ela_conf, ela_details = classify_with_ela(metrics)
    if ela_pred != "DESCONOCIDO":
        vote_ela = 1 if ela_pred == "GENERADA POR IA" else 0
        votes.append((vote_ela, ela_conf * weights["ela"]))
        details["ela"] = {"prediction": ela_pred, "confidence": ela_conf, **ela_details}
    
    # Voto Laplaciano (usando configuración existente)
    lap_score = metrics.get("laplacian_score")
    if lap_score is not None:
        # Umbral del sistema existente
        lap_threshold = 4.5853
        vote_lap = 1 if lap_score > lap_threshold else 0
        lap_conf = min(95.0, 60.0 + abs(lap_score - lap_threshold) * 10)
        votes.append((vote_lap, lap_conf * weights["laplacian"]))
        details["laplacian"] = {
            "prediction": "GENERADA POR IA" if vote_lap else "REAL",
            "confidence": round(lap_conf, 2),
            "value": lap_score,
            "threshold": lap_threshold
        }
    
    # Voto FFT (ratio alta/baja frecuencia)
    fft_ratio = metrics.get("fft_ratio_af_bf") or metrics.get("ratio_af_bf")
    if fft_ratio is not None:
        # Umbral heurístico para FFT
        fft_threshold = 50.0
        vote_fft = 1 if fft_ratio > fft_threshold else 0
        fft_conf = min(90.0, 55.0 + abs(fft_ratio - fft_threshold) * 0.5)
        votes.append((vote_fft, fft_conf * weights["fft"]))
        details["fft"] = {
            "prediction": "GENERADA POR IA" if vote_fft else "REAL",
            "confidence": round(fft_conf, 2),
            "value": fft_ratio,
            "threshold": fft_threshold
        }
    
    if not votes:
        return "DESCONOCIDO", 0.0, {"reason": "Sin métricas disponibles"}
    
    # Votación ponderada
    total_weight = sum(v[1] for v in votes)
    weighted_vote = sum(v[0] * v[1] for v in votes) / total_weight
    
    # Decisión final
    is_fake = weighted_vote > 0.5
    prediction = "GENERADA POR IA" if is_fake else "REAL"
    
    # Confianza basada en qué tan decisiva fue la votación
    vote_strength = abs(weighted_vote - 0.5) * 2  # 0-1
    confidence = 60.0 + vote_strength * 35.0
    
    details["voting"] = {
        "weighted_vote": round(weighted_vote, 3),
        "vote_strength": round(vote_strength, 3),
        "num_classifiers": len(votes)
    }
    
    return prediction, round(confidence, 2), details


def update_ela_config(feature: str, threshold: float, direction: str):
    """
    Actualiza configuración de clasificador ELA.
    
    Args:
        feature: Nombre de la métrica a usar
        threshold: Valor umbral
        direction: 'higher_is_fake' o 'lower_is_fake'
    """
    global ELA_CONFIG
    ELA_CONFIG["primary_feature"] = feature
    ELA_CONFIG["threshold"] = threshold
    ELA_CONFIG["direction"] = direction
    print(f"Configuración ELA actualizada: {feature} @ {threshold} ({direction})")
