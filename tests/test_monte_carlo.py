"""
Tests unitaires pour le module Monte Carlo.

Ce module teste les fonctionnalités des algorithmes :
- MonteCarloES
- OnPolicyFirstVisitMC
- OffPolicyMC
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, MagicMock

from src.monte_carlo import MonteCarloES, OnPolicyFirstVisitMC, OffPolicyMC


class TestMonteCarloES:
    """Tests pour la classe MonteCarloES."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.env = Mock()
        self.env.observation_space = Mock()
        self.env.observation_space.n = 10  # Nombre d'états
        self.env.action_space = Mock()
        self.env.action_space.n = 4  # Nombre d'actions
        
        self.mc_es = MonteCarloES(
            env=self.env,
            gamma=0.9
        )
    
    def test_init(self):
        """Test de l'initialisation."""
        assert self.mc_es.env == self.env
        assert self.mc_es.gamma == 0.9
        assert self.mc_es.Q.shape == (10, 4)  # nS x nA
        assert len(self.mc_es.returns) == 0
        assert self.mc_es.policy is None
        assert self.mc_es.history == []
    
    def test_generate_episode_stub(self):
        """Test du stub de generate_episode."""
        result = self.mc_es.generate_episode()
        assert result is None
    
    def test_update_q_values_stub(self):
        """Test du stub de update_q_values."""
        episode = [((0, 0), 0, 1), ((0, 1), 1, 0)]
        result = self.mc_es.update_q_values(episode)
        assert result is None
    
    def test_improve_policy_stub(self):
        """Test du stub de improve_policy."""
        result = self.mc_es.improve_policy()
        assert result is None
    
    def test_train_stub(self):
        """Test du stub de train."""
        result = self.mc_es.train(num_episodes=10)
        assert result is None
    
    def test_save_load(self):
        """Test de sauvegarde et chargement."""
        # Configurer des données fictives
        self.mc_es.Q[(0, 0)] = np.array([1, 2, 3, 4])
        self.mc_es.policy = {0: 1, 1: 2}
        self.mc_es.history = [{'episode': 1, 'reward': 10}]
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Sauvegarder
            self.mc_es.save(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Créer une nouvelle instance et charger
            new_mc_es = MonteCarloES(self.env)
            new_mc_es.load(tmp_path)
            
            # Vérifier que les données sont correctes
            np.testing.assert_array_equal(new_mc_es.Q[(0, 0)], [1, 2, 3, 4])
            assert new_mc_es.policy == {0: 1, 1: 2}
            assert new_mc_es.gamma == 0.9
            assert new_mc_es.epsilon == 0.1
            assert new_mc_es.history == [{'episode': 1, 'reward': 10}]
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestOnPolicyFirstVisitMC:
    """Tests pour la classe OnPolicyFirstVisitMC."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.env = Mock()
        self.env.observation_space = Mock()
        self.env.observation_space.n = 10  # Nombre d'états
        self.env.action_space = Mock()
        self.env.action_space.n = 4  # Nombre d'actions
        
        self.on_policy_mc = OnPolicyFirstVisitMC(
            env=self.env,
            gamma=0.9,
            epsilon=0.1
        )
    
    def test_init(self):
        """Test de l'initialisation."""
        assert self.on_policy_mc.env == self.env
        assert self.on_policy_mc.gamma == 0.9
        assert self.on_policy_mc.epsilon == 0.1
        assert self.on_policy_mc.Q.shape == (10, 4)  # nS x nA
        assert len(self.on_policy_mc.returns) == 0
        assert self.on_policy_mc.policy.shape == (10, 4)  # nS x nA (politique stochastique)
        assert self.on_policy_mc.history == []
    
    def test_epsilon_greedy_policy_stub(self):
        """Test du stub de epsilon_greedy_policy."""
        result = self.on_policy_mc.epsilon_greedy_policy(0)
        assert result is None
    
    def test_generate_episode_stub(self):
        """Test du stub de generate_episode."""
        result = self.on_policy_mc.generate_episode()
        assert result is None
    
    def test_update_q_values_stub(self):
        """Test du stub de update_q_values."""
        episode = [((0, 0), 0, 1), ((0, 1), 1, 0)]
        result = self.on_policy_mc.update_q_values(episode)
        assert result is None
    
    def test_train_stub(self):
        """Test du stub de train."""
        result = self.on_policy_mc.train(num_episodes=10)
        assert result is None


class TestOffPolicyMC:
    """Tests pour la classe OffPolicyMC."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.env = Mock()
        self.env.observation_space = Mock()
        self.env.observation_space.n = 10  # Nombre d'états
        self.env.action_space = Mock()
        self.env.action_space.n = 4  # Nombre d'actions
        
        self.off_policy_mc = OffPolicyMC(
            env=self.env,
            gamma=0.9
        )
    
    def test_init(self):
        """Test de l'initialisation."""
        assert self.off_policy_mc.env == self.env
        assert self.off_policy_mc.gamma == 0.9
        assert self.off_policy_mc.Q.shape == (10, 4)  # nS x nA
        assert self.off_policy_mc.C.shape == (10, 4)  # nS x nA
        assert self.off_policy_mc.target_policy.shape == (10,)  # nS (déterministe)
        assert self.off_policy_mc.behavior_policy.shape == (10, 4)  # nS x nA (stochastique)
        assert self.off_policy_mc.history == []
    
    def test_behavior_policy_action_stub(self):
        """Test du stub de behavior_policy_action."""
        result = self.off_policy_mc.behavior_policy_action(0)
        assert result is None
    
    def test_target_policy_action_stub(self):
        """Test du stub de target_policy_action."""
        result = self.off_policy_mc.target_policy_action(0)
        assert result is None
    
    def test_calculate_importance_ratio_stub(self):
        """Test du stub de calculate_importance_ratio."""
        episode = [((0, 0), 0, 1), ((0, 1), 1, 0)]
        result = self.off_policy_mc.calculate_importance_ratio(episode)
        assert result is None
    
    def test_generate_episode_stub(self):
        """Test du stub de generate_episode."""
        result = self.off_policy_mc.generate_episode()
        assert result is None
    
    def test_update_q_values_stub(self):
        """Test du stub de update_q_values."""
        episode = [((0, 0), 0, 1), ((0, 1), 1, 0)]
        result = self.off_policy_mc.update_q_values(episode)
        assert result is None
    
    def test_train_stub(self):
        """Test du stub de train."""
        result = self.off_policy_mc.train(num_episodes=10)
        assert result is None
    
    def test_save_load(self):
        """Test de sauvegarde et chargement."""
        # Configurer des données fictives
        self.off_policy_mc.Q[(0, 0)] = np.array([1, 2, 3, 4])
        self.off_policy_mc.C[(0, 0)] = np.array([0.1, 0.2, 0.3, 0.4])
        self.off_policy_mc.target_policy = {0: 1, 1: 2}
        self.off_policy_mc.behavior_policy = {0: 0, 1: 1}
        self.off_policy_mc.history = [{'episode': 1, 'reward': 10}]
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Sauvegarder
            self.off_policy_mc.save(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Créer une nouvelle instance et charger
            new_off_policy_mc = OffPolicyMC(self.env)
            new_off_policy_mc.load(tmp_path)
            
            # Vérifier que les données sont correctes
            np.testing.assert_array_equal(new_off_policy_mc.Q[(0, 0)], [1, 2, 3, 4])
            np.testing.assert_array_equal(new_off_policy_mc.C[(0, 0)], [0.1, 0.2, 0.3, 0.4])
            assert new_off_policy_mc.target_policy == {0: 1, 1: 2}
            assert new_off_policy_mc.behavior_policy == {0: 0, 1: 1}
            assert new_off_policy_mc.gamma == 0.9
            assert new_off_policy_mc.epsilon == 0.1
            assert new_off_policy_mc.history == [{'episode': 1, 'reward': 10}]
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestIntegration:
    """Tests d'intégration pour les algorithmes Monte Carlo."""
    
    def test_algorithms_have_same_interface(self):
        """Test que les algorithmes ont une interface similaire."""
        env = Mock()
        env.observation_space = Mock()
        env.observation_space.n = 10
        env.action_space = Mock()
        env.action_space.n = 4
        
        mc_es = MonteCarloES(env)
        on_policy = OnPolicyFirstVisitMC(env)
        off_policy = OffPolicyMC(env)
        
        algorithms = [mc_es, on_policy, off_policy]
        
        # Vérifier que les méthodes principales existent
        for algo in algorithms:
            assert hasattr(algo, 'train')
            assert hasattr(algo, 'save')
            assert hasattr(algo, 'load')
            assert hasattr(algo, 'Q')
            assert hasattr(algo, 'history')
    
    def test_different_epsilon_values(self):
        """Test avec différentes valeurs d'epsilon."""
        env = Mock()
        env.observation_space = Mock()
        env.observation_space.n = 10
        env.action_space = Mock()
        env.action_space.n = 4
        
        # MonteCarloES n'a pas de paramètre epsilon
        mc_09 = MonteCarloES(env, gamma=0.9)
        mc_099 = MonteCarloES(env, gamma=0.99)
        
        assert mc_09.gamma == 0.9
        assert mc_099.gamma == 0.99
    
    def test_different_gamma_values(self):
        """Test avec différentes valeurs de gamma."""
        env = Mock()
        env.observation_space = Mock()
        env.observation_space.n = 10
        env.action_space = Mock()
        env.action_space.n = 4
        
        mc_09 = OnPolicyFirstVisitMC(env, gamma=0.9)
        mc_099 = OnPolicyFirstVisitMC(env, gamma=0.99)
        
        assert mc_09.gamma == 0.9
        assert mc_099.gamma == 0.99


if __name__ == '__main__':
    pytest.main([__file__]) 