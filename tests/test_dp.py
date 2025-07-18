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
        self.env.nS = 16  # Nombre d'états
        self.env.nA = 4   # Nombre d'actions
        
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
    
    def test_policy_evaluation_implementation(self):
        """Test de l'implémentation de policy_evaluation."""
        # Créer un environnement simple
        self.policy_iteration._initialize_mdp_structures()
        
        # Initialiser une politique simple
        policy = np.zeros(self.policy_iteration.n_states, dtype=int)
        
        # Initialiser V
        self.policy_iteration.V = np.zeros(self.policy_iteration.n_states)
        
        # Tester policy_evaluation
        result = self.policy_iteration.policy_evaluation(policy)
        
        # Vérifier que le résultat est un array NumPy
        assert isinstance(result, np.ndarray)
        assert result.shape == (self.policy_iteration.n_states,)
    
    def test_policy_improvement_implementation(self):
        """Test de l'implémentation de policy_improvement."""
        # Créer un environnement simple
        self.policy_iteration._initialize_mdp_structures()
        
        # Initialiser une fonction de valeur
        V = np.random.random(self.policy_iteration.n_states)
        
        # Initialiser une politique
        self.policy_iteration.policy = np.zeros(self.policy_iteration.n_states, dtype=int)
        
        # Tester policy_improvement
        new_policy, policy_stable = self.policy_iteration.policy_improvement(V)
        
        # Vérifier les types de retour
        assert isinstance(new_policy, np.ndarray)
        assert isinstance(policy_stable, bool)
        assert new_policy.shape == (self.policy_iteration.n_states,)
    
    def test_train_implementation(self):
        """Test de l'implémentation de train."""
        # Tester avec un nombre limité d'itérations
        result = self.policy_iteration.train(max_iterations=5)
        
        # Vérifier que le résultat est un dictionnaire
        assert isinstance(result, dict)
        
        # Vérifier les clés attendues
        expected_keys = ['policy', 'V', 'history', 'iterations', 'converged']
        for key in expected_keys:
            assert key in result
        
        # Vérifier les types
        assert isinstance(result['policy'], np.ndarray)
        assert isinstance(result['V'], np.ndarray)
        assert isinstance(result['history'], list)
        assert isinstance(result['iterations'], int)
        assert isinstance(result['converged'], bool)
    
    def test_save_load(self):
        """Test de sauvegarde et chargement."""
        # Initialiser les structures
        self.policy_iteration._initialize_mdp_structures()
        
        # Configurer des données fictives
        self.policy_iteration.V = np.array([1.0, 2.0, 3.0, 4.0])
        self.policy_iteration.policy = np.array([0, 1, 2, 3])
        self.policy_iteration.history = [{'iteration': 1, 'avg_value': 2.5, 'max_value': 4.0, 'policy_stable': True}]
        
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
            np.testing.assert_array_equal(new_policy_iteration.V, [1.0, 2.0, 3.0, 4.0])
            np.testing.assert_array_equal(new_policy_iteration.policy, [0, 1, 2, 3])
            assert new_policy_iteration.gamma == 0.9
            assert new_policy_iteration.theta == 1e-6
            assert new_policy_iteration.history == [{'iteration': 1, 'avg_value': 2.5, 'max_value': 4.0, 'policy_stable': True}]
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_initialize_mdp_structures(self):
        """Test de l'initialisation des structures MDP."""
        self.policy_iteration._initialize_mdp_structures()
        
        # Vérifier que les attributs sont créés
        assert hasattr(self.policy_iteration, 'n_states')
        assert hasattr(self.policy_iteration, 'n_actions')
        assert hasattr(self.policy_iteration, 'states')
        assert hasattr(self.policy_iteration, 'actions')
        assert hasattr(self.policy_iteration, 'rewards')
        assert hasattr(self.policy_iteration, 'transition_probs')
        
        # Vérifier les types
        assert isinstance(self.policy_iteration.n_states, int)
        assert isinstance(self.policy_iteration.n_actions, int)
        assert isinstance(self.policy_iteration.states, list)
        assert isinstance(self.policy_iteration.actions, list)
        assert isinstance(self.policy_iteration.rewards, list)
        assert isinstance(self.policy_iteration.transition_probs, np.ndarray)
        
        # Vérifier les dimensions
        expected_shape = (self.policy_iteration.n_states, 
                         self.policy_iteration.n_actions, 
                         self.policy_iteration.n_states, 
                         len(self.policy_iteration.rewards))
        assert self.policy_iteration.transition_probs.shape == expected_shape


class TestValueIteration:
    """Tests pour la classe ValueIteration."""
    
    def setup_method(self):
        """Configuration pour chaque test."""
        self.env = Mock()
        self.env.observation_space = Mock()
        self.env.action_space = Mock()
        self.env.action_space.n = 4
        self.env.observation_space.n = 16
        self.env.nS = 16  # Nombre d'états
        self.env.nA = 4   # Nombre d'actions
        
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
    
    def test_value_update_implementation(self):
        """Test de l'implémentation de value_update."""
        # Initialiser les structures MDP
        self.value_iteration._initialize_mdp_structures()
        
        # Initialiser une fonction de valeur
        V = np.zeros(self.value_iteration.n_states)
        
        # Tester value_update
        new_V, delta = self.value_iteration.value_update(V)
        
        # Vérifier les types de retour
        assert isinstance(new_V, np.ndarray)
        assert isinstance(delta, float)
        assert new_V.shape == V.shape
        assert delta >= 0
    
    def test_extract_policy_implementation(self):
        """Test de l'implémentation de extract_policy."""
        # Initialiser les structures MDP
        self.value_iteration._initialize_mdp_structures()
        
        # Initialiser une fonction de valeur
        V = np.random.random(self.value_iteration.n_states)
        
        # Tester extract_policy
        policy = self.value_iteration.extract_policy(V)
        
        # Vérifier le type et la forme
        assert isinstance(policy, np.ndarray)
        assert policy.shape == (self.value_iteration.n_states, self.value_iteration.n_actions)
        
        # Vérifier que chaque ligne est une distribution de probabilité
        for s in range(self.value_iteration.n_states):
            if s not in self.value_iteration.terminal_states:
                assert np.isclose(np.sum(policy[s]), 1.0), f"Ligne {s} n'est pas une distribution de probabilité"
                assert np.all(policy[s] >= 0), f"Ligne {s} contient des probabilités négatives"
    
    def test_train_implementation(self):
        """Test de l'implémentation de train."""
        # Tester avec un nombre limité d'itérations
        result = self.value_iteration.train(max_iterations=5)
        
        # Vérifier que le résultat est un dictionnaire
        assert isinstance(result, dict)
        
        # Vérifier les clés attendues
        expected_keys = ['V', 'policy', 'history', 'iterations', 'converged', 'final_delta']
        for key in expected_keys:
            assert key in result
        
        # Vérifier les types
        assert isinstance(result['V'], np.ndarray)
        assert isinstance(result['policy'], np.ndarray)
        assert isinstance(result['history'], list)
        assert isinstance(result['iterations'], int)
        assert isinstance(result['converged'], bool)
        assert isinstance(result['final_delta'], (float, type(None)))
        
        # Vérifier les dimensions
        assert result['V'].shape == (self.value_iteration.n_states,)
        assert result['policy'].shape == (self.value_iteration.n_states, self.value_iteration.n_actions)
    
    def test_convergence_simple_case(self):
        """Test de convergence sur un cas simple."""
        # Créer un MDP simple
        class SimpleMDP:
            def __init__(self):
                self.nS = 2
                self.nA = 1
                self.P = {
                    0: {0: [(1.0, 1, 1.0, True)]},
                    1: {0: [(1.0, 1, 0.0, True)]}
                }
        
        simple_env = SimpleMDP()
        vi = ValueIteration(simple_env, gamma=0.9, theta=1e-6)
        result = vi.train(max_iterations=100)
        
        # Vérifier la convergence
        assert result['converged'], "L'algorithme devrait converger sur ce cas simple"
        
        # Vérifier la fonction de valeur
        # V[0] = 1 + 0.9 * V[1] = 1 + 0.9 * 0 = 1
        # V[1] = 0 (état terminal)
        assert abs(result['V'][0] - 1.0) < 1e-4, f"V[0] devrait être 1.0, obtenu: {result['V'][0]}"
        assert abs(result['V'][1] - 0.0) < 1e-4, f"V[1] devrait être 0.0, obtenu: {result['V'][1]}"
    
    def test_save_load(self):
        """Test de sauvegarde et chargement."""
        # Initialiser les structures
        self.value_iteration._initialize_mdp_structures()
        
        # Configurer des données fictives
        self.value_iteration.V = np.array([1.0, 2.0, 3.0, 4.0])
        self.value_iteration.policy = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], 
                                               [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        self.value_iteration.history = [{'iteration': 1, 'delta': 0.5, 'avg_value': 2.5, 'max_value': 4.0, 'converged': False}]
        
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
            np.testing.assert_array_equal(new_value_iteration.V, [1.0, 2.0, 3.0, 4.0])
            np.testing.assert_array_equal(new_value_iteration.policy, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], 
                                                                       [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
            assert new_value_iteration.gamma == 0.9
            assert new_value_iteration.theta == 1e-6
            assert new_value_iteration.history == [{'iteration': 1, 'delta': 0.5, 'avg_value': 2.5, 'max_value': 4.0, 'converged': False}]
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_initialize_mdp_structures(self):
        """Test de l'initialisation des structures MDP."""
        self.value_iteration._initialize_mdp_structures()
        
        # Vérifier que les attributs sont créés
        assert hasattr(self.value_iteration, 'n_states')
        assert hasattr(self.value_iteration, 'n_actions')
        assert hasattr(self.value_iteration, 'states')
        assert hasattr(self.value_iteration, 'actions')
        assert hasattr(self.value_iteration, 'rewards')
        assert hasattr(self.value_iteration, 'transition_probs')
        assert hasattr(self.value_iteration, 'terminal_states')
        
        # Vérifier les types
        assert isinstance(self.value_iteration.n_states, int)
        assert isinstance(self.value_iteration.n_actions, int)
        assert isinstance(self.value_iteration.states, list)
        assert isinstance(self.value_iteration.actions, list)
        assert isinstance(self.value_iteration.rewards, list)
        assert isinstance(self.value_iteration.transition_probs, np.ndarray)
        assert isinstance(self.value_iteration.terminal_states, list)
        
        # Vérifier les dimensions
        expected_shape = (self.value_iteration.n_states, 
                         self.value_iteration.n_actions, 
                         self.value_iteration.n_states, 
                         len(self.value_iteration.rewards))
        assert self.value_iteration.transition_probs.shape == expected_shape


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