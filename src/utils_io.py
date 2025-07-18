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
    
    try:
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
    except Exception as e:
        raise IOError(f"Erreur lors de la sauvegarde: {e}")


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
    
    try:
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
    except Exception as e:
        raise IOError(f"Erreur lors du chargement: {e}")


def _detect_format(filepath: str) -> str:
    """
    Détecte automatiquement le format basé sur l'extension du fichier.
    
    Args:
        filepath: Chemin du fichier
    
    Returns:
        Format détecté
    """
    _, ext = os.path.splitext(filepath.lower())
    
    if ext == '.pkl' or ext == '.pickle':
        return 'pickle'
    elif ext == '.joblib' or ext == '.jbl':
        return 'joblib'
    elif ext == '.json':
        return 'json'
    elif ext == '.npz':
        return 'npz'
    else:
        # Par défaut, essayer pickle
        return 'pickle'


def _save_pickle(data: Dict[str, Any], filepath: str) -> None:
    """
    Sauvegarde avec pickle.
    
    Args:
        data: Données à sauvegarder
        filepath: Chemin de sauvegarde
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(filepath: str) -> Dict[str, Any]:
    """
    Charge avec pickle.
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        Données chargées
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def _save_joblib(data: Dict[str, Any], filepath: str) -> None:
    """
    Sauvegarde avec joblib.
    
    Args:
        data: Données à sauvegarder
        filepath: Chemin de sauvegarde
    """
    if not JOBLIB_AVAILABLE:
        raise ImportError("joblib n'est pas disponible. Utilisez pickle à la place.")
    
    joblib.dump(data, filepath, compress=3)


def _load_joblib(filepath: str) -> Dict[str, Any]:
    """
    Charge avec joblib.
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        Données chargées
    """
    if not JOBLIB_AVAILABLE:
        raise ImportError("joblib n'est pas disponible. Utilisez pickle à la place.")
    
    return joblib.load(filepath)


def _save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Sauvegarde avec JSON.
    
    Args:
        data: Données à sauvegarder
        filepath: Chemin de sauvegarde
    """
    # Convertir les arrays NumPy en listes pour la sérialisation JSON
    json_data = _convert_numpy_to_json(data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def _load_json(filepath: str) -> Dict[str, Any]:
    """
    Charge avec JSON.
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        Données chargées
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Convertir les listes en arrays NumPy si nécessaire
    return _convert_json_to_numpy(json_data)


def _save_npz(data: Dict[str, Any], filepath: str) -> None:
    """
    Sauvegarde avec NumPy NPZ.
    
    Args:
        data: Données à sauvegarder
        filepath: Chemin de sauvegarde
    """
    # Séparer les arrays NumPy des autres données
    arrays = {}
    other_data = {}
    
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            other_data[key] = value
    
    # Sauvegarder les arrays
    if arrays:
        np.savez_compressed(filepath, **arrays)
    
    # Sauvegarder les autres données avec pickle si nécessaire
    if other_data:
        metadata_path = filepath.replace('.npz', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(other_data, f)


def _load_npz(filepath: str) -> Dict[str, Any]:
    """
    Charge avec NumPy NPZ.
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        Données chargées
    """
    data = {}
    
    # Charger les arrays
    if os.path.exists(filepath):
        with np.load(filepath) as npz_file:
            for key in npz_file.files:
                data[key] = npz_file[key]
    
    # Charger les métadonnées si elles existent
    metadata_path = filepath.replace('.npz', '_metadata.pkl')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            data.update(metadata)
    
    return data


def _convert_numpy_to_json(obj: Any) -> Any:
    """
    Convertit récursivement les objets NumPy en types JSON sérialisables.
    
    Args:
        obj: Objet à convertir
        
    Returns:
        Objet converti
    """
    if isinstance(obj, np.ndarray):
        return {
            '__numpy_array__': True,
            'data': obj.tolist(),
            'dtype': str(obj.dtype),
            'shape': obj.shape
        }
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: _convert_numpy_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_to_json(item) for item in obj)
    else:
        return obj


def _convert_json_to_numpy(obj: Any) -> Any:
    """
    Convertit récursivement les objets JSON en objets NumPy.
    
    Args:
        obj: Objet à convertir
        
    Returns:
        Objet converti
    """
    if isinstance(obj, dict):
        if '__numpy_array__' in obj and obj['__numpy_array__']:
            # Reconstituer l'array NumPy
            return np.array(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])
        else:
            return {key: _convert_json_to_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_json_to_numpy(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_json_to_numpy(item) for item in obj)
    else:
        return obj


def create_model_directory(base_path: str, model_name: str) -> str:
    """
    Crée un dossier pour sauvegarder les modèles.
    
    Args:
        base_path: Chemin de base
        model_name: Nom du modèle
        
    Returns:
        Chemin du dossier créé
    """
    model_dir = os.path.join(base_path, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def get_model_info(filepath: str) -> Dict[str, Any]:
    """
    Récupère les informations d'un modèle sauvegardé.
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        Informations du modèle
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} n'existe pas.")
    
    info = {
        'filepath': filepath,
        'format': _detect_format(filepath),
        'size': os.path.getsize(filepath),
        'modified': os.path.getmtime(filepath)
    }
    
    try:
        # Essayer de charger le modèle pour obtenir plus d'informations
        model_data = load_model(filepath)
        info['keys'] = list(model_data.keys())
        
        # Informations sur les arrays NumPy
        numpy_info = {}
        for key, value in model_data.items():
            if isinstance(value, np.ndarray):
                numpy_info[key] = {
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'size': value.size
                }
        
        if numpy_info:
            info['numpy_arrays'] = numpy_info
            
    except Exception as e:
        info['error'] = str(e)
    
    return info


def cleanup_model_files(directory: str, keep_recent: int = 5) -> None:
    """
    Nettoie les anciens fichiers de modèles.
    
    Args:
        directory: Dossier à nettoyer
        keep_recent: Nombre de fichiers récents à conserver
    """
    if not os.path.exists(directory):
        return
    
    # Récupérer tous les fichiers de modèles
    model_files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and any(filename.endswith(ext) for ext in ['.pkl', '.pickle', '.joblib', '.json', '.npz']):
            model_files.append((filepath, os.path.getmtime(filepath)))
    
    # Trier par date de modification (plus récent en premier)
    model_files.sort(key=lambda x: x[1], reverse=True)
    
    # Supprimer les anciens fichiers
    for filepath, _ in model_files[keep_recent:]:
        try:
            os.remove(filepath)
            print(f"Fichier supprimé: {filepath}")
            
            # Supprimer aussi le fichier de métadonnées s'il existe
            if filepath.endswith('.npz'):
                metadata_path = filepath.replace('.npz', '_metadata.pkl')
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    print(f"Métadonnées supprimées: {metadata_path}")
                    
        except Exception as e:
            print(f"Erreur lors de la suppression de {filepath}: {e}")


def backup_model(filepath: str, backup_dir: Optional[str] = None) -> str:
    """
    Crée une sauvegarde d'un modèle.
    
    Args:
        filepath: Chemin du fichier original
        backup_dir: Dossier de sauvegarde (optionnel)
        
    Returns:
        Chemin de la sauvegarde
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} n'existe pas.")
    
    if backup_dir is None:
        backup_dir = os.path.join(os.path.dirname(filepath), 'backups')
    
    os.makedirs(backup_dir, exist_ok=True)
    
    # Créer un nom de fichier unique avec timestamp
    import time
    timestamp = int(time.time())
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    backup_filename = f"{name}_{timestamp}{ext}"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    # Copier le fichier
    import shutil
    shutil.copy2(filepath, backup_path)
    
    # Copier aussi les métadonnées si elles existent
    if filepath.endswith('.npz'):
        metadata_path = filepath.replace('.npz', '_metadata.pkl')
        if os.path.exists(metadata_path):
            backup_metadata_path = backup_path.replace('.npz', '_metadata.pkl')
            shutil.copy2(metadata_path, backup_metadata_path)
    
    return backup_path 