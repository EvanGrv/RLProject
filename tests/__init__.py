"""
Package de tests pour les algorithmes de Reinforcement Learning.

Ce package contient tous les tests unitaires pour valider le bon fonctionnement
des algorithmes implémentés.
"""

# Configuration commune pour les tests
import pytest
import numpy as np
from unittest.mock import Mock


def create_mock_env(action_space_n=4, observation_space_n=16):
    """
    Crée un environnement mock pour les tests.
    
    Args:
        action_space_n: Nombre d'actions disponibles
        observation_space_n: Nombre d'états observables
    
    Returns:
        Mock environment pour les tests
    """
    env = Mock()
    env.action_space = Mock()
    env.action_space.n = action_space_n
    env.observation_space = Mock()
    env.observation_space.n = observation_space_n
    
    return env


def assert_algorithm_interface(algorithm):
    """
    Vérifie qu'un algorithme a l'interface attendue.
    
    Args:
        algorithm: Instance d'algorithme à tester
    """
    # Vérifier les méthodes obligatoires
    assert hasattr(algorithm, 'train'), "L'algorithme doit avoir une méthode 'train'"
    assert hasattr(algorithm, 'save'), "L'algorithme doit avoir une méthode 'save'"
    assert hasattr(algorithm, 'load'), "L'algorithme doit avoir une méthode 'load'"
    
    # Vérifier les attributs obligatoires
    assert hasattr(algorithm, 'history'), "L'algorithme doit avoir un attribut 'history'"
    assert hasattr(algorithm, 'env'), "L'algorithme doit avoir un attribut 'env'"
    
    # Vérifier que les méthodes sont callables
    assert callable(algorithm.train), "La méthode 'train' doit être callable"
    assert callable(algorithm.save), "La méthode 'save' doit être callable"
    assert callable(algorithm.load), "La méthode 'load' doit être callable"


# Configuration par défaut pour les tests
DEFAULT_TEST_CONFIG = {
    'action_space_n': 4,
    'observation_space_n': 16,
    'num_episodes': 10,
    'num_iterations': 100,
    'tolerance': 1e-6,
    'seed': 42
}


# Fixtures communes pour les tests
@pytest.fixture
def mock_env():
    """Fixture pour créer un environnement mock."""
    return create_mock_env()


@pytest.fixture
def test_config():
    """Fixture pour la configuration de test."""
    return DEFAULT_TEST_CONFIG.copy()


@pytest.fixture
def random_seed():
    """Fixture pour initialiser la seed aléatoire."""
    np.random.seed(DEFAULT_TEST_CONFIG['seed'])
    return DEFAULT_TEST_CONFIG['seed']


# Constantes utiles pour les tests
TEST_DATA_DIR = "tests/data/"
TEST_MODELS_DIR = "tests/models/"
TEST_RESULTS_DIR = "tests/results/"

# Marques pour les tests
SLOW_TESTS = pytest.mark.slow
INTEGRATION_TESTS = pytest.mark.integration
UNIT_TESTS = pytest.mark.unit 