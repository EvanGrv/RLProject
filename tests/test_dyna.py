"""
Tests unitaires pour le module Dyna.

Ce module teste les fonctionnalités des algorithmes :
- DynaQ
- DynaQPlus
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, MagicMock

from src.dyna import DynaQ, DynaQPlus


class TestDynaQ:
    """Tests pour la classe DynaQ."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.env = Mock()
        self.env.action_space = Mock()
        self.env.action_space.n = 4
        
        self.dyna_q = DynaQ(
            env=self.env,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1,
            n_planning=5
        )
    
    def test_init(self):
        """Test de l'initialisation."""
        assert self.dyna_q.env == self.env
        assert self.dyna_q.alpha == 0.1
        assert self.dyna_q.gamma == 0.9
        assert self.dyna_q.epsilon == 0.1
        assert self.dyna_q.n_planning == 5
        assert len(self.dyna_q.Q) == 0
        assert len(self.dyna_q.model) == 0
        assert len(self.dyna_q.visited_states) == 0
        assert self.dyna_q.policy is None
        assert self.dyna_q.history == []
    
    def test_epsilon_greedy_policy_stub(self):
        """Test du stub de epsilon_greedy_policy."""
        result = self.dyna_q.epsilon_greedy_policy(0)
        assert result is None
    
    def test_update_model_stub(self):
        """Test du stub de update_model."""
        result = self.dyna_q.update_model(0, 1, 1, 1.0)
        assert result is None
    
    def test_update_q_value_stub(self):
        """Test du stub de update_q_value."""
        result = self.dyna_q.update_q_value(0, 1, 1.0, 1)
        assert result is None
    
    def test_planning_step_stub(self):
        """Test du stub de planning_step."""
        result = self.dyna_q.planning_step()
        assert result is None
    
    def test_train_step_stub(self):
        """Test du stub de train_step."""
        result = self.dyna_q.train_step(0, 1, 1.0, 1)
        assert result is None
    
    def test_train_episode_stub(self):
        """Test du stub de train_episode."""
        result = self.dyna_q.train_episode()
        assert result is None
    
    def test_train_stub(self):
        """Test du stub de train."""
        result = self.dyna_q.train(num_episodes=10)
        assert result is None
    
    def test_save_load(self):
        """Test de sauvegarde et chargement."""
        # Configurer des données fictives
        self.dyna_q.Q[(0, 0)] = np.array([1, 2, 3, 4])
        self.dyna_q.model = {0: {1: (1, 1.0)}}
        self.dyna_q.policy = {0: 1, 1: 2}
        self.dyna_q.history = [{'episode': 1, 'reward': 10}]
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Sauvegarder
            self.dyna_q.save(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Créer une nouvelle instance et charger
            new_dyna_q = DynaQ(self.env)
            new_dyna_q.load(tmp_path)
            
            # Vérifier que les données sont correctes
            np.testing.assert_array_equal(new_dyna_q.Q[(0, 0)], [1, 2, 3, 4])
            assert new_dyna_q.model == {0: {1: (1, 1.0)}}
            assert new_dyna_q.policy == {0: 1, 1: 2}
            assert new_dyna_q.alpha == 0.1
            assert new_dyna_q.gamma == 0.9
            assert new_dyna_q.epsilon == 0.1
            assert new_dyna_q.n_planning == 5
            assert new_dyna_q.history == [{'episode': 1, 'reward': 10}]
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestDynaQPlus:
    """Tests pour la classe DynaQPlus."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.env = Mock()
        self.env.action_space = Mock()
        self.env.action_space.n = 4
        
        self.dyna_q_plus = DynaQPlus(
            env=self.env,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.1,
            n_planning=5,
            kappa=0.001
        )
    
    def test_init(self):
        """Test de l'initialisation."""
        assert self.dyna_q_plus.env == self.env
        assert self.dyna_q_plus.alpha == 0.1
        assert self.dyna_q_plus.gamma == 0.9
        assert self.dyna_q_plus.epsilon == 0.1
        assert self.dyna_q_plus.n_planning == 5
        assert self.dyna_q_plus.kappa == 0.001
        assert len(self.dyna_q_plus.Q) == 0
        assert len(self.dyna_q_plus.model) == 0
        assert len(self.dyna_q_plus.time_since_visit) == 0
        assert self.dyna_q_plus.current_time == 0
        assert self.dyna_q_plus.policy is None
        assert self.dyna_q_plus.history == []
    
    def test_epsilon_greedy_policy_stub(self):
        """Test du stub de epsilon_greedy_policy."""
        result = self.dyna_q_plus.epsilon_greedy_policy(0)
        assert result is None
    
    def test_update_model_stub(self):
        """Test du stub de update_model."""
        result = self.dyna_q_plus.update_model(0, 1, 1, 1.0)
        assert result is None
    
    def test_exploration_bonus_stub(self):
        """Test du stub de exploration_bonus."""
        result = self.dyna_q_plus.exploration_bonus(0, 1)
        assert result is None
    
    def test_update_q_value_stub(self):
        """Test du stub de update_q_value."""
        result = self.dyna_q_plus.update_q_value(0, 1, 1.0, 1)
        assert result is None
    
    def test_planning_step_stub(self):
        """Test du stub de planning_step."""
        result = self.dyna_q_plus.planning_step()
        assert result is None
    
    def test_train_step_stub(self):
        """Test du stub de train_step."""
        result = self.dyna_q_plus.train_step(0, 1, 1.0, 1)
        assert result is None
    
    def test_train_episode_stub(self):
        """Test du stub de train_episode."""
        result = self.dyna_q_plus.train_episode()
        assert result is None
    
    def test_train_stub(self):
        """Test du stub de train."""
        result = self.dyna_q_plus.train(num_episodes=10)
        assert result is None
    
    def test_save_load(self):
        """Test de sauvegarde et chargement."""
        # Configurer des données fictives
        self.dyna_q_plus.Q[(0, 0)] = np.array([1, 2, 3, 4])
        self.dyna_q_plus.model = {0: {1: (1, 1.0)}}
        self.dyna_q_plus.time_since_visit = {0: {1: 5}}
        self.dyna_q_plus.current_time = 100
        self.dyna_q_plus.policy = {0: 1, 1: 2}
        self.dyna_q_plus.history = [{'episode': 1, 'reward': 10}]
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Sauvegarder
            self.dyna_q_plus.save(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Créer une nouvelle instance et charger
            new_dyna_q_plus = DynaQPlus(self.env)
            new_dyna_q_plus.load(tmp_path)
            
            # Vérifier que les données sont correctes
            np.testing.assert_array_equal(new_dyna_q_plus.Q[(0, 0)], [1, 2, 3, 4])
            assert new_dyna_q_plus.model == {0: {1: (1, 1.0)}}
            assert new_dyna_q_plus.current_time == 100
            assert new_dyna_q_plus.policy == {0: 1, 1: 2}
            assert new_dyna_q_plus.alpha == 0.1
            assert new_dyna_q_plus.gamma == 0.9
            assert new_dyna_q_plus.epsilon == 0.1
            assert new_dyna_q_plus.n_planning == 5
            assert new_dyna_q_plus.kappa == 0.001
            assert new_dyna_q_plus.history == [{'episode': 1, 'reward': 10}]
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestIntegration:
    """Tests d'intégration pour les algorithmes Dyna."""
    
    def test_algorithms_have_same_interface(self):
        """Test que les algorithmes ont une interface similaire."""
        env = Mock()
        env.action_space = Mock()
        env.action_space.n = 4
        
        dyna_q = DynaQ(env)
        dyna_q_plus = DynaQPlus(env)
        
        algorithms = [dyna_q, dyna_q_plus]
        
        # Vérifier que les méthodes principales existent
        for algo in algorithms:
            assert hasattr(algo, 'train')
            assert hasattr(algo, 'train_episode')
            assert hasattr(algo, 'train_step')
            assert hasattr(algo, 'planning_step')
            assert hasattr(algo, 'save')
            assert hasattr(algo, 'load')
            assert hasattr(algo, 'Q')
            assert hasattr(algo, 'model')
            assert hasattr(algo, 'history')
            assert hasattr(algo, 'epsilon_greedy_policy')
    
    def test_different_n_planning_values(self):
        """Test avec différentes valeurs de n_planning."""
        env = Mock()
        env.action_space = Mock()
        env.action_space.n = 4
        
        dyna_5 = DynaQ(env, n_planning=5)
        dyna_10 = DynaQ(env, n_planning=10)
        
        assert dyna_5.n_planning == 5
        assert dyna_10.n_planning == 10
    
    def test_different_kappa_values(self):
        """Test avec différentes valeurs de kappa pour Dyna-Q+."""
        env = Mock()
        env.action_space = Mock()
        env.action_space.n = 4
        
        dyna_001 = DynaQPlus(env, kappa=0.001)
        dyna_01 = DynaQPlus(env, kappa=0.01)
        
        assert dyna_001.kappa == 0.001
        assert dyna_01.kappa == 0.01
    
    def test_dyna_q_plus_has_additional_attributes(self):
        """Test que Dyna-Q+ a les attributs supplémentaires."""
        env = Mock()
        env.action_space = Mock()
        env.action_space.n = 4
        
        dyna_q = DynaQ(env)
        dyna_q_plus = DynaQPlus(env)
        
        # Dyna-Q+ a des attributs supplémentaires
        assert hasattr(dyna_q_plus, 'kappa')
        assert hasattr(dyna_q_plus, 'time_since_visit')
        assert hasattr(dyna_q_plus, 'current_time')
        assert hasattr(dyna_q_plus, 'exploration_bonus')
        
        # Dyna-Q n'a pas ces attributs
        assert not hasattr(dyna_q, 'kappa')
        assert not hasattr(dyna_q, 'time_since_visit')
        assert not hasattr(dyna_q, 'current_time')
        assert not hasattr(dyna_q, 'exploration_bonus')


if __name__ == '__main__':
    pytest.main([__file__]) 