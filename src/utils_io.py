"""
Module utilitaire pour la sauvegarde et le chargement de modèles.

Ce module fournit des fonctions pour sauvegarder et charger des modèles
de reinforcement learning avec différents formats :
- pickle (par défaut)
- joblib (optimisé pour les arrays NumPy)
- JSON (pour les données sérialisables)
- NPZ (format NumPy compressé)
"""

import pickle
import json
import os
import numpy as np
from typing import Any, Dict, Union, Optional
import warnings

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    warnings.warn("joblib n'est pas disponible. Utilisation de pickle par défaut.")


def save_model(data: Dict[str, Any], filepath: str, format: str = 'pickle') -> None:
    """
    Sauvegarde un modèle dans le format spécifié.
    
    Args:
        data: Dictionnaire contenant les données du modèle
        filepath: Chemin de sauvegarde
        format: Format de sauvegarde ('pickle', 'joblib', 'json', 'npz')
    
    Raises:
        ValueError: Si le format n'est pas supporté
        IOError: Si l'écriture échoue
    """
    # Créer le dossier si nécessaire
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == 'pickle':
        _save_pickle(data, filepath)
    elif format == 'joblib':
        _save_joblib(data, filepath)
    elif format == 'json':
        _save_json(data, filepath)
    elif format == 'npz':
        _save_npz(data, filepath)
    else:
        raise ValueError(f"Format '{format}' non supporté. "
                        "Formats disponibles: pickle, joblib, json, npz")


def load_model(filepath: str, format: str = 'auto') -> Dict[str, Any]:
    """
    Charge un modèle depuis le format spécifié.
    
    Args:
        filepath: Chemin du fichier à charger
        format: Format de chargement ('auto', 'pickle', 'joblib', 'json', 'npz')
    
    Returns:
        Dictionnaire contenant les données du modèle
    
    Raises:
        ValueError: Si le format n'est pas supporté
        FileNotFoundError: Si le fichier n'existe pas
        IOError: Si la lecture échoue
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} n'existe pas.")
    
    if format == 'auto':
        format = _detect_format(filepath)
    
    if format == 'pickle':
        return _load_pickle(filepath)
    elif format == 'joblib':
        return _load_joblib(filepath)
    elif format == 'json':
        return _load_json(filepath)
    elif format == 'npz':
        return _load_npz(filepath)
    else:
        raise ValueError(f"Format '{format}' non supporté. "
                        "Formats disponibles: pickle, joblib, json, npz")


def _detect_format(filepath: str) -> str:
    """
    Détecte automatiquement le format basé sur l'extension du fichier.
    
    Args:
        filepath: Chemin du fichier
    
    Returns:
        Format détecté
    """
    extension = os.path.splitext(filepath)[1].lower()
    
    if extension == '.pkl':
        return 'pickle'
    elif extension == '.joblib':
        return 'joblib'
    elif extension == '.json':
        return 'json'
    elif extension == '.npz':
        return 'npz'
    else:
        # Par défaut, essayer pickle
        return 'pickle'


def _save_pickle(data: Dict[str, Any], filepath: str) -> None:
    """Sauvegarde avec pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def _load_pickle(filepath: str) -> Dict[str, Any]:
    """Charge avec pickle."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def _save_joblib(data: Dict[str, Any], filepath: str) -> None:
    """Sauvegarde avec joblib."""
    if not JOBLIB_AVAILABLE:
        raise ImportError("joblib n'est pas disponible. Installez-le avec: pip install joblib")
    
    joblib.dump(data, filepath)


def _load_joblib(filepath: str) -> Dict[str, Any]:
    """Charge avec joblib."""
    if not JOBLIB_AVAILABLE:
        raise ImportError("joblib n'est pas disponible. Installez-le avec: pip install joblib")
    
    return joblib.load(filepath)


def _save_json(data: Dict[str, Any], filepath: str) -> None:
    """Sauvegarde avec JSON (seulement pour les données sérialisables)."""
    # Convertir les arrays NumPy en listes
    json_data = _convert_numpy_to_json(data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def _load_json(filepath: str) -> Dict[str, Any]:
    """Charge avec JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convertir les listes en arrays NumPy si nécessaire
    return _convert_json_to_numpy(data)


def _save_npz(data: Dict[str, Any], filepath: str) -> None:
    """Sauvegarde avec NPZ (seulement pour les arrays NumPy)."""
    # Séparer les arrays NumPy des autres données
    arrays = {}
    metadata = {}
    
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            metadata[key] = value
    
    # Sauvegarder les arrays
    if arrays:
        np.savez_compressed(filepath, **arrays)
    
    # Sauvegarder les métadonnées séparément
    if metadata:
        metadata_path = filepath.replace('.npz', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


def _load_npz(filepath: str) -> Dict[str, Any]:
    """Charge avec NPZ."""
    data = {}
    
    # Charger les arrays
    if os.path.exists(filepath):
        with np.load(filepath) as npz_file:
            for key in npz_file.files:
                data[key] = npz_file[key]
    
    # Charger les métadonnées
    metadata_path = filepath.replace('.npz', '_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            data.update(metadata)
    
    return data


def _convert_numpy_to_json(data: Any) -> Any:
    """Convertit récursivement les arrays NumPy en listes pour JSON."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: _convert_numpy_to_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_convert_numpy_to_json(item) for item in data]
    elif isinstance(data, (np.int64, np.int32, np.int16, np.int8)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32, np.float16)):
        return float(data)
    else:
        return data


def _convert_json_to_numpy(data: Any) -> Any:
    """Convertit récursivement les listes en arrays NumPy si approprié."""
    if isinstance(data, dict):
        return {key: _convert_json_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Essayer de convertir en array NumPy si c'est une liste de nombres
        try:
            return np.array(data)
        except (ValueError, TypeError):
            # Si la conversion échoue, retourner la liste originale
            return [_convert_json_to_numpy(item) for item in data]
    else:
        return data


def list_saved_models(directory: str = 'models/', format: Optional[str] = None) -> list:
    """
    Liste tous les modèles sauvegardés dans un dossier.
    
    Args:
        directory: Dossier à explorer
        format: Format spécifique à chercher (None pour tous)
    
    Returns:
        Liste des fichiers de modèles trouvés
    """
    if not os.path.exists(directory):
        return []
    
    files = []
    extensions = {
        'pickle': ['.pkl', '.pickle'],
        'joblib': ['.joblib'],
        'json': ['.json'],
        'npz': ['.npz']
    }
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if format is None:
                # Tous les formats
                for ext_list in extensions.values():
                    if any(filename.endswith(ext) for ext in ext_list):
                        files.append(filepath)
                        break
            else:
                # Format spécifique
                if format in extensions:
                    if any(filename.endswith(ext) for ext in extensions[format]):
                        files.append(filepath)
    
    return sorted(files)


def get_model_info(filepath: str) -> Dict[str, Any]:
    """
    Récupère les informations sur un modèle sauvegardé.
    
    Args:
        filepath: Chemin du fichier de modèle
    
    Returns:
        Dictionnaire contenant les informations du modèle
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} n'existe pas.")
    
    info = {
        'filepath': filepath,
        'size': os.path.getsize(filepath),
        'modified': os.path.getmtime(filepath),
        'format': _detect_format(filepath)
    }
    
    try:
        # Essayer de charger et extraire des informations basiques
        data = load_model(filepath)
        info['keys'] = list(data.keys()) if isinstance(data, dict) else []
        info['loadable'] = True
    except Exception as e:
        info['keys'] = []
        info['loadable'] = False
        info['error'] = str(e)
    
    return info 