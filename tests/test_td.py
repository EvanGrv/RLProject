"""
Tests unitaires pour le module Temporal Difference.

Ce module teste les fonctionnalités des algorithmes :
- Sarsa
- QLearning
- ExpectedSarsa
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, MagicMock

from src.td import Sarsa, QLearning, ExpectedSarsa


class TestSarsa:
    """Tests pour la classe Sarsa."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.env = Mock()
        self.env.observation_space = Mock()
        self.env.observation_space.n = 10  # Nombre d'états
        self.env.action_space = Mock()
        self.env.action_space.n = 4  # Nombre d'actions
        
        self.sarsa = Sarsa(
            env=self.env,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1
        )
    
    def test_init(self):
        """Test de l'initialisation."""
        assert self.sarsa.env == self.env
        assert self.sarsa.alpha == 0.1
        assert self.sarsa.gamma == 0.9
        assert self.sarsa.epsilon == 0.1
        assert self.sarsa.Q.shape == (10, 4)  # nS x nA
        assert self.sarsa.policy is None
        assert self.sarsa.history == []
    
    def test_epsilon_greedy_policy_stub(self):
        """Test du stub de epsilon_greedy_policy."""
        result = self.sarsa.epsilon_greedy_policy(0)
        assert result is None
    
    def test_update_q_value_stub(self):
        """Test du stub de update_q_value."""
        result = self.sarsa.update_q_value(0, 1, 1.0, 1, 2)
        assert result is None
    
    def test_train_episode_stub(self):
        """Test du stub de train_episode."""
        result = self.sarsa.train_episode()
        assert result is None
    
    def test_train_stub(self):
        """Test du stub de train."""
        result = self.sarsa.train(num_episodes=10)
        assert result is None
    
    def test_save_load(self):
        """Test de sauvegarde et chargement."""
        # Configurer des données fictives
        self.sarsa.Q[(0, 0)] = np.array([1, 2, 3, 4])
        self.sarsa.policy = {0: 1, 1: 2}
        self.sarsa.history = [{'episode': 1, 'reward': 10}]
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Sauvegarder
            self.sarsa.save(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Créer une nouvelle instance et charger
            new_sarsa = Sarsa(self.env)
            new_sarsa.load(tmp_path)
            
            # Vérifier que les données sont correctes
            np.testing.assert_array_equal(new_sarsa.Q[(0, 0)], [1, 2, 3, 4])
            assert new_sarsa.policy == {0: 1, 1: 2}
            assert new_sarsa.alpha == 0.1
            assert new_sarsa.gamma == 0.9
            assert new_sarsa.epsilon == 0.1
            assert new_sarsa.history == [{'episode': 1, 'reward': 10}]
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestQLearning:
    """Tests pour la classe QLearning."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.env = Mock()
        self.env.observation_space = Mock()
        self.env.observation_space.n = 10  # Nombre d'états
        self.env.action_space = Mock()
        self.env.action_space.n = 4  # Nombre d'actions
        
        self.q_learning = QLearning(
            env=self.env,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1
        )
    
    def test_init(self):
        """Test de l'initialisation."""
        assert self.q_learning.env == self.env
        assert self.q_learning.alpha == 0.1
        assert self.q_learning.gamma == 0.9
        assert self.q_learning.epsilon == 0.1
        assert self.q_learning.Q.shape == (10, 4)  # nS x nA
        assert self.q_learning.policy is None
        assert self.q_learning.history == []
    
    def test_epsilon_greedy_policy_stub(self):
        """Test du stub de epsilon_greedy_policy."""
        result = self.q_learning.epsilon_greedy_policy(0)
        assert result is None
    
    def test_greedy_policy_stub(self):
        """Test du stub de greedy_policy."""
        result = self.q_learning.greedy_policy(0)
        assert result is None
    
    def test_update_q_value_stub(self):
        """Test du stub de update_q_value."""
        result = self.q_learning.update_q_value(0, 1, 1.0, 1)
        assert result is None
    
    def test_train_episode_stub(self):
        """Test du stub de train_episode."""
        result = self.q_learning.train_episode()
        assert result is None
    
    def test_train_stub(self):
        """Test du stub de train."""
        result = self.q_learning.train(num_episodes=10)
        assert result is None


class TestExpectedSarsa:
    """Tests pour la classe ExpectedSarsa."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.env = Mock()
        self.env.observation_space = Mock()
        self.env.observation_space.n = 10  # Nombre d'états
        self.env.action_space = Mock()
        self.env.action_space.n = 4  # Nombre d'actions
        
        self.expected_sarsa = ExpectedSarsa(
            env=self.env,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1
        )
    
    def test_init(self):
        """Test de l'initialisation."""
        assert self.expected_sarsa.env == self.env
        assert self.expected_sarsa.alpha == 0.1
        assert self.expected_sarsa.gamma == 0.9
        assert self.expected_sarsa.epsilon == 0.1
        assert self.expected_sarsa.Q.shape == (10, 4)  # nS x nA
        assert self.expected_sarsa.policy is None
        assert self.expected_sarsa.history == []
    
    def test_epsilon_greedy_policy_stub(self):
        """Test du stub de epsilon_greedy_policy."""
        result = self.expected_sarsa.epsilon_greedy_policy(0)
        assert result is None
    
    def test_expected_q_value_stub(self):
        """Test du stub de expected_q_value."""
        result = self.expected_sarsa.expected_q_value(0)
        assert result is None
    
    def test_update_q_value_stub(self):
        """Test du stub de update_q_value."""
        result = self.expected_sarsa.update_q_value(0, 1, 1.0, 1)
        assert result is None
    
    def test_train_episode_stub(self):
        """Test du stub de train_episode."""
        result = self.expected_sarsa.train_episode()
        assert result is None
    
    def test_train_stub(self):
        """Test du stub de train."""
        result = self.expected_sarsa.train(num_episodes=10)
        assert result is None


class TestIntegration:
    """Tests d'intégration pour les algorithmes TD."""
    
    def test_algorithms_have_same_interface(self):
        """Test que les algorithmes ont une interface similaire."""
        env = Mock()
        env.observation_space = Mock()
        env.observation_space.n = 10
        env.action_space = Mock()
        env.action_space.n = 4
        
        sarsa = Sarsa(env)
        q_learning = QLearning(env)
        expected_sarsa = ExpectedSarsa(env)
        
        algorithms = [sarsa, q_learning, expected_sarsa]
        
        # Vérifier que les méthodes principales existent
        for algo in algorithms:
            assert hasattr(algo, 'train')
            assert hasattr(algo, 'train_episode')
            assert hasattr(algo, 'save')
            assert hasattr(algo, 'load')
            assert hasattr(algo, 'Q')
            assert hasattr(algo, 'history')
            assert hasattr(algo, 'epsilon_greedy_policy')
    
    def test_different_alpha_values(self):
        """Test avec différentes valeurs d'alpha."""
        env = Mock()
        env.observation_space = Mock()
        env.observation_space.n = 10
        env.action_space = Mock()
        env.action_space.n = 4
        
        sarsa_01 = Sarsa(env, alpha=0.1)
        sarsa_05 = Sarsa(env, alpha=0.5)
        
        assert sarsa_01.alpha == 0.1
        assert sarsa_05.alpha == 0.5
    
    def test_different_epsilon_values(self):
        """Test avec différentes valeurs d'epsilon."""
        env = Mock()
        env.observation_space = Mock()
        env.observation_space.n = 10
        env.action_space = Mock()
        env.action_space.n = 4
        
        q_01 = QLearning(env, epsilon=0.1)
        q_05 = QLearning(env, epsilon=0.5)
        
        assert q_01.epsilon == 0.1
        assert q_05.epsilon == 0.5
    
    def test_different_gamma_values(self):
        """Test avec différentes valeurs de gamma."""
        env = Mock()
        env.observation_space = Mock()
        env.observation_space.n = 10
        env.action_space = Mock()
        env.action_space.n = 4
        
        es_09 = ExpectedSarsa(env, gamma=0.9)
        es_099 = ExpectedSarsa(env, gamma=0.99)
        
        assert es_09.gamma == 0.9
        assert es_099.gamma == 0.99


if __name__ == '__main__':
    pytest.main([__file__]) 