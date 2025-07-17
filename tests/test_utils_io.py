"""
Tests unitaires pour le module utils_io.

Ce module teste les fonctionnalités de sauvegarde et chargement :
- save_model / load_model
- Différents formats (pickle, joblib, json, npz)
- Fonctions utilitaires
"""

import pytest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch

from src.utils_io import (
    save_model, load_model, list_saved_models, get_model_info,
    _detect_format, _convert_numpy_to_json, _convert_json_to_numpy
)


class TestSaveLoadModel:
    """Tests pour save_model et load_model."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.test_data = {
            'Q': np.array([[1, 2], [3, 4]]),
            'policy': np.array([0, 1]),
            'gamma': 0.9,
            'epsilon': 0.1,
            'history': [{'episode': 1, 'reward': 10}]
        }
    
    def test_save_load_pickle(self):
        """Test de sauvegarde et chargement avec pickle."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Sauvegarder
            save_model(self.test_data, tmp_path, format='pickle')
            assert os.path.exists(tmp_path)
            
            # Charger
            loaded_data = load_model(tmp_path, format='pickle')
            
            # Vérifier
            np.testing.assert_array_equal(loaded_data['Q'], self.test_data['Q'])
            np.testing.assert_array_equal(loaded_data['policy'], self.test_data['policy'])
            assert loaded_data['gamma'] == self.test_data['gamma']
            assert loaded_data['epsilon'] == self.test_data['epsilon']
            assert loaded_data['history'] == self.test_data['history']
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_save_load_json(self):
        """Test de sauvegarde et chargement avec JSON."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Sauvegarder
            save_model(self.test_data, tmp_path, format='json')
            assert os.path.exists(tmp_path)
            
            # Charger
            loaded_data = load_model(tmp_path, format='json')
            
            # Vérifier (les arrays NumPy sont convertis en listes puis reconvertis)
            np.testing.assert_array_equal(loaded_data['Q'], self.test_data['Q'])
            np.testing.assert_array_equal(loaded_data['policy'], self.test_data['policy'])
            assert loaded_data['gamma'] == self.test_data['gamma']
            assert loaded_data['epsilon'] == self.test_data['epsilon']
            assert loaded_data['history'] == self.test_data['history']
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    @patch('src.utils_io.JOBLIB_AVAILABLE', True)
    def test_save_load_joblib(self):
        """Test de sauvegarde et chargement avec joblib."""
        with patch('src.utils_io.joblib') as mock_joblib:
            mock_joblib.dump = Mock()
            mock_joblib.load = Mock(return_value=self.test_data)
            
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                # Sauvegarder
                save_model(self.test_data, tmp_path, format='joblib')
                mock_joblib.dump.assert_called_once_with(self.test_data, tmp_path)
                
                # Charger
                loaded_data = load_model(tmp_path, format='joblib')
                mock_joblib.load.assert_called_once_with(tmp_path)
                
                assert loaded_data == self.test_data
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    def test_save_load_npz(self):
        """Test de sauvegarde et chargement avec NPZ."""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name
            metadata_path = tmp_path.replace('.npz', '_metadata.json')
        
        try:
            # Sauvegarder
            save_model(self.test_data, tmp_path, format='npz')
            assert os.path.exists(tmp_path)
            assert os.path.exists(metadata_path)
            
            # Charger
            loaded_data = load_model(tmp_path, format='npz')
            
            # Vérifier
            np.testing.assert_array_equal(loaded_data['Q'], self.test_data['Q'])
            np.testing.assert_array_equal(loaded_data['policy'], self.test_data['policy'])
            assert loaded_data['gamma'] == self.test_data['gamma']
            assert loaded_data['epsilon'] == self.test_data['epsilon']
            assert loaded_data['history'] == self.test_data['history']
            
        finally:
            for path in [tmp_path, metadata_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def test_auto_format_detection(self):
        """Test de détection automatique du format."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Sauvegarder avec pickle
            save_model(self.test_data, tmp_path, format='pickle')
            
            # Charger avec auto-détection
            loaded_data = load_model(tmp_path, format='auto')
            
            # Vérifier
            np.testing.assert_array_equal(loaded_data['Q'], self.test_data['Q'])
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_unsupported_format_error(self):
        """Test d'erreur pour format non supporté."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValueError, match="Format 'invalid' non supporté"):
                save_model(self.test_data, tmp_path, format='invalid')
                
            with pytest.raises(ValueError, match="Format 'invalid' non supporté"):
                load_model(tmp_path, format='invalid')
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_file_not_found_error(self):
        """Test d'erreur pour fichier inexistant."""
        with pytest.raises(FileNotFoundError):
            load_model('non_existent_file.pkl')


class TestUtilityFunctions:
    """Tests pour les fonctions utilitaires."""
    
    def test_detect_format(self):
        """Test de détection de format."""
        assert _detect_format('model.pkl') == 'pickle'
        assert _detect_format('model.joblib') == 'joblib'
        assert _detect_format('model.json') == 'json'
        assert _detect_format('model.npz') == 'npz'
        assert _detect_format('model.unknown') == 'pickle'  # fallback
    
    def test_convert_numpy_to_json(self):
        """Test de conversion NumPy vers JSON."""
        data = {
            'array': np.array([1, 2, 3]),
            'int': np.int64(42),
            'float': np.float64(3.14),
            'nested': {'inner_array': np.array([[1, 2], [3, 4]])},
            'list': [np.array([1, 2]), 'string']
        }
        
        result = _convert_numpy_to_json(data)
        
        assert result['array'] == [1, 2, 3]
        assert result['int'] == 42
        assert result['float'] == 3.14
        assert result['nested']['inner_array'] == [[1, 2], [3, 4]]
        assert result['list'][0] == [1, 2]
        assert result['list'][1] == 'string'
    
    def test_convert_json_to_numpy(self):
        """Test de conversion JSON vers NumPy."""
        data = {
            'array': [1, 2, 3],
            'int': 42,
            'float': 3.14,
            'nested': {'inner_array': [[1, 2], [3, 4]]},
            'list': [[1, 2], 'string']
        }
        
        result = _convert_json_to_numpy(data)
        
        assert isinstance(result['array'], np.ndarray)
        np.testing.assert_array_equal(result['array'], [1, 2, 3])
        assert result['int'] == 42
        assert result['float'] == 3.14
        assert isinstance(result['nested']['inner_array'], np.ndarray)
        np.testing.assert_array_equal(result['nested']['inner_array'], [[1, 2], [3, 4]])
    
    def test_list_saved_models(self):
        """Test de listage des modèles sauvegardés."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Créer quelques fichiers de test
            test_files = [
                'model1.pkl',
                'model2.json',
                'model3.joblib',
                'model4.npz',
                'not_a_model.txt'
            ]
            
            for filename in test_files:
                filepath = os.path.join(temp_dir, filename)
                with open(filepath, 'w') as f:
                    f.write('test')
            
            # Tester sans filtre de format
            all_models = list_saved_models(temp_dir)
            assert len(all_models) == 4  # Exclut .txt
            
            # Tester avec filtre pickle
            pickle_models = list_saved_models(temp_dir, format='pickle')
            assert len(pickle_models) == 1
            assert 'model1.pkl' in pickle_models[0]
            
            # Tester avec filtre JSON
            json_models = list_saved_models(temp_dir, format='json')
            assert len(json_models) == 1
            assert 'model2.json' in json_models[0]
    
    def test_get_model_info(self):
        """Test de récupération d'informations sur un modèle."""
        test_data = {'key': 'value'}
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Sauvegarder un modèle
            save_model(test_data, tmp_path)
            
            # Récupérer les informations
            info = get_model_info(tmp_path)
            
            assert info['filepath'] == tmp_path
            assert info['size'] > 0
            assert info['format'] == 'pickle'
            assert info['loadable'] is True
            assert 'key' in info['keys']
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_get_model_info_corrupted_file(self):
        """Test d'informations sur un fichier corrompu."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
            # Écrire des données non-pickle
            tmp.write(b'corrupted data')
        
        try:
            info = get_model_info(tmp_path)
            
            assert info['filepath'] == tmp_path
            assert info['size'] > 0
            assert info['format'] == 'pickle'
            assert info['loadable'] is False
            assert 'error' in info
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_get_model_info_nonexistent_file(self):
        """Test d'informations sur un fichier inexistant."""
        with pytest.raises(FileNotFoundError):
            get_model_info('nonexistent.pkl')


class TestDirectoryCreation:
    """Tests pour la création automatique de dossiers."""
    
    def test_save_creates_directory(self):
        """Test que save_model crée les dossiers nécessaires."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, 'nested', 'folder', 'model.pkl')
            
            # Le dossier nested/folder n'existe pas
            assert not os.path.exists(os.path.dirname(nested_path))
            
            # Sauvegarder doit créer le dossier
            save_model({'test': 'data'}, nested_path)
            
            # Vérifier que le dossier et le fichier existent
            assert os.path.exists(os.path.dirname(nested_path))
            assert os.path.exists(nested_path)


if __name__ == '__main__':
    pytest.main([__file__]) 