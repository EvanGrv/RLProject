"""
Tests unitaires pour le module Dynamic Programming.

Ce module teste les fonctionnalités des algorithmes :
- PolicyIteration
- ValueIteration
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, MagicMock

from src.dp import PolicyIteration, ValueIteration


class TestPolicyIteration:
    """Tests pour la classe PolicyIteration."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.env = Mock()
        self.env.observation_space = Mock()
        self.env.action_space = Mock()
        self.env.action_space.n = 4
        self.env.observation_space.n = 16
        
        self.policy_iteration = PolicyIteration(
            env=self.env,
            gamma=0.9,
            theta=1e-6
        )
    
    def test_init(self):
        """Test de l'initialisation."""
        assert self.policy_iteration.env == self.env
        assert self.policy_iteration.gamma == 0.9
        assert self.policy_iteration.theta == 1e-6
        assert self.policy_iteration.V is None
        assert self.policy_iteration.policy is None
        assert self.policy_iteration.history == []
    
    def test_policy_evaluation_stub(self):
        """Test du stub de policy_evaluation."""
        policy = np.ones((16, 4)) / 4  # Politique uniforme
        result = self.policy_iteration.policy_evaluation(policy)
        # Le stub retourne None
        assert result is None
    
    def test_policy_improvement_stub(self):
        """Test du stub de policy_improvement."""
        V = np.zeros(16)
        result = self.policy_iteration.policy_improvement(V)
        # Le stub retourne None
        assert result is None
    
    def test_train_stub(self):
        """Test du stub de train."""
        result = self.policy_iteration.train(max_iterations=10)
        # Le stub retourne None
        assert result is None
    
    def test_save_load(self):
        """Test de sauvegarde et chargement."""
        # Configurer des données fictives
        self.policy_iteration.V = np.array([1, 2, 3, 4])
        self.policy_iteration.policy = np.array([0, 1, 2, 3])
        self.policy_iteration.history = [{'episode': 1, 'reward': 10}]
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Sauvegarder
            self.policy_iteration.save(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Créer une nouvelle instance et charger
            new_policy_iteration = PolicyIteration(self.env)
            new_policy_iteration.load(tmp_path)
            
            # Vérifier que les données sont correctes
            np.testing.assert_array_equal(new_policy_iteration.V, [1, 2, 3, 4])
            np.testing.assert_array_equal(new_policy_iteration.policy, [0, 1, 2, 3])
            assert new_policy_iteration.gamma == 0.9
            assert new_policy_iteration.theta == 1e-6
            assert new_policy_iteration.history == [{'episode': 1, 'reward': 10}]
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestValueIteration:
    """Tests pour la classe ValueIteration."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.env = Mock()
        self.env.observation_space = Mock()
        self.env.action_space = Mock()
        self.env.action_space.n = 4
        self.env.observation_space.n = 16
        
        self.value_iteration = ValueIteration(
            env=self.env,
            gamma=0.9,
            theta=1e-6
        )
    
    def test_init(self):
        """Test de l'initialisation."""
        assert self.value_iteration.env == self.env
        assert self.value_iteration.gamma == 0.9
        assert self.value_iteration.theta == 1e-6
        assert self.value_iteration.V is None
        assert self.value_iteration.policy is None
        assert self.value_iteration.history == []
    
    def test_value_update_stub(self):
        """Test du stub de value_update."""
        V = np.zeros(16)
        result = self.value_iteration.value_update(V)
        # Le stub retourne None
        assert result is None
    
    def test_extract_policy_stub(self):
        """Test du stub de extract_policy."""
        V = np.zeros(16)
        result = self.value_iteration.extract_policy(V)
        # Le stub retourne None
        assert result is None
    
    def test_train_stub(self):
        """Test du stub de train."""
        result = self.value_iteration.train(max_iterations=10)
        # Le stub retourne None
        assert result is None
    
    def test_save_load(self):
        """Test de sauvegarde et chargement."""
        # Configurer des données fictives
        self.value_iteration.V = np.array([1, 2, 3, 4])
        self.value_iteration.policy = np.array([0, 1, 2, 3])
        self.value_iteration.history = [{'episode': 1, 'reward': 10}]
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Sauvegarder
            self.value_iteration.save(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Créer une nouvelle instance et charger
            new_value_iteration = ValueIteration(self.env)
            new_value_iteration.load(tmp_path)
            
            # Vérifier que les données sont correctes
            np.testing.assert_array_equal(new_value_iteration.V, [1, 2, 3, 4])
            np.testing.assert_array_equal(new_value_iteration.policy, [0, 1, 2, 3])
            assert new_value_iteration.gamma == 0.9
            assert new_value_iteration.theta == 1e-6
            assert new_value_iteration.history == [{'episode': 1, 'reward': 10}]
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestIntegration:
    """Tests d'intégration pour les algorithmes DP."""
    
    def test_algorithms_have_same_interface(self):
        """Test que les deux algorithmes ont la même interface."""
        env = Mock()
        env.action_space = Mock()
        env.action_space.n = 4
        
        pi = PolicyIteration(env)
        vi = ValueIteration(env)
        
        # Vérifier que les méthodes principales existent
        assert hasattr(pi, 'train')
        assert hasattr(vi, 'train')
        assert hasattr(pi, 'save')
        assert hasattr(vi, 'save')
        assert hasattr(pi, 'load')
        assert hasattr(vi, 'load')
        
        # Vérifier que les attributs principaux existent
        assert hasattr(pi, 'V')
        assert hasattr(vi, 'V')
        assert hasattr(pi, 'policy')
        assert hasattr(vi, 'policy')
        assert hasattr(pi, 'history')
        assert hasattr(vi, 'history')
    
    def test_different_gamma_values(self):
        """Test avec différentes valeurs de gamma."""
        env = Mock()
        env.action_space = Mock()
        env.action_space.n = 4
        
        pi_09 = PolicyIteration(env, gamma=0.9)
        pi_099 = PolicyIteration(env, gamma=0.99)
        
        assert pi_09.gamma == 0.9
        assert pi_099.gamma == 0.99
    
    def test_different_theta_values(self):
        """Test avec différentes valeurs de theta."""
        env = Mock()
        env.action_space = Mock()
        env.action_space.n = 4
        
        vi_1e6 = ValueIteration(env, theta=1e-6)
        vi_1e8 = ValueIteration(env, theta=1e-8)
        
        assert vi_1e6.theta == 1e-6
        assert vi_1e8.theta == 1e-8


if __name__ == '__main__':
    pytest.main([__file__]) 