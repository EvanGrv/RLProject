"""
Module de Dynamic Programming pour le Reinforcement Learning.

Ce module implémente les algorithmes de programmation dynamique :
- Policy Iteration (Itération de politique)
- Value Iteration (Itération de valeur)

Ces algorithmes sont utilisés pour résoudre des MDPs (Markov Decision Processes)
lorsque la dynamique de l'environnement est connue.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from src.utils_io import save_model, load_model


class PolicyIteration:
    """
    Implémentation de l'algorithme Policy Iteration.
    
    Policy Iteration alterne entre l'évaluation de politique et l'amélioration
    de politique jusqu'à convergence vers la politique optimale.
    """
    
    def __init__(self, env: Any, gamma: float = 0.9, theta: float = 1e-6):
        """
        Initialise Policy Iteration.
        
        Args:
            env: Environnement MDP
            gamma: Facteur de discount
            theta: Seuil de convergence pour l'évaluation de politique
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = None
        self.policy = None
        self.history = []
        
        # Initialiser les structures MDP
        self._initialize_mdp_structures()
        
    def _initialize_mdp_structures(self):
        """Initialise les structures MDP nécessaires."""
        # Obtenir les informations MDP de l'environnement
        if hasattr(self.env, 'get_mdp_info'):
            # Environnement personnalisé avec get_mdp_info()
            mdp_info = self.env.get_mdp_info()
            self.states = list(mdp_info['states'])
            self.actions = mdp_info['actions']
            self.transition_matrix = mdp_info['transition_matrix']
            self.reward_matrix = mdp_info['reward_matrix']
            self.terminals = mdp_info['terminals']
            self.n_states = len(self.states)
            self.n_actions = len(self.actions)
            
        elif hasattr(self.env, 'nS') and hasattr(self.env, 'nA'):
            # Environnement gym discret
            self.n_states = self.env.nS
            self.n_actions = self.env.nA
            self.states = list(range(self.n_states))
            self.actions = list(range(self.n_actions))
            self.terminals = []
            
            # Extraire les matrices de transition et de récompense
            if hasattr(self.env, 'P'):
                self.transition_matrix, self.reward_matrix = self._extract_gym_dynamics()
            else:
                raise ValueError("Environnement gym sans dynamique disponible")
                
        else:
            # Environnement générique
            self.n_states = getattr(self.env, 'observation_space', type('obj', (object,), {'n': 16})).n
            self.n_actions = getattr(self.env, 'action_space', type('obj', (object,), {'n': 4})).n
            self.states = list(range(self.n_states))
            self.actions = list(range(self.n_actions))
            self.terminals = []
            
            # Créer des matrices par défaut
            self.transition_matrix = np.zeros((self.n_states, self.n_actions, self.n_states))
            self.reward_matrix = np.zeros((self.n_states, self.n_actions))
            
    def _extract_gym_dynamics(self):
        """Extrait les dynamiques d'un environnement Gym."""
        # Matrice de transition P(s'|s,a)
        transition_matrix = np.zeros((self.n_states, self.n_actions, self.n_states))
        # Matrice de récompense R(s,a)
        reward_matrix = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if s in self.env.P and a in self.env.P[s]:
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        transition_matrix[s, a, next_state] += prob
                        reward_matrix[s, a] += prob * reward
                        
                        if done and next_state not in self.terminals:
                            self.terminals.append(next_state)
        
        return transition_matrix, reward_matrix
        
    def policy_evaluation(self, policy: np.ndarray) -> np.ndarray:
        """
        Évalue une politique donnée.
        
        Args:
            policy: Politique à évaluer (actions pour chaque état)
            
        Returns:
            Fonction de valeur d'état
        """
        V = np.zeros(self.n_states)
        
        while True:
            delta = 0.0
            
            for s in self.states:
                if s in self.terminals:
                    continue
                    
                v = V[s]
                action = policy[s]
                
                # Calculer la nouvelle valeur
                new_value = self.reward_matrix[s, action]
                for s_prime in self.states:
                    new_value += self.gamma * self.transition_matrix[s, action, s_prime] * V[s_prime]
                
                V[s] = new_value
                delta = max(delta, abs(v - V[s]))
            
            if delta < self.theta:
                break
        
        return V
        
    def policy_improvement(self, V: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Améliore la politique basée sur la fonction de valeur.
        
        Args:
            V: Fonction de valeur d'état
            
        Returns:
            Tuple (nouvelle politique, politique stable)
        """
        new_policy = np.copy(self.policy)
        policy_stable = True
        
        for s in self.states:
            if s in self.terminals:
                continue
                
            old_action = new_policy[s]
            best_action = None
            best_value = -np.inf
            
            # Évaluer toutes les actions possibles
            for a in self.actions:
                value = self.reward_matrix[s, a]
                for s_prime in self.states:
                    value += self.gamma * self.transition_matrix[s, a, s_prime] * V[s_prime]
                
                if value > best_value:
                    best_value = value
                    best_action = a
            
            if best_action != old_action:
                policy_stable = False
            
            new_policy[s] = best_action
        
        return new_policy, policy_stable
        
    def policy_iteration(self, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exécute l'algorithme Policy Iteration complet.
        
        Args:
            max_iterations: Nombre maximum d'itérations
            
        Returns:
            Tuple (politique optimale, fonction de valeur optimale)
        """
        # Initialiser la politique aléatoirement
        self.policy = np.random.choice(self.actions, size=self.n_states)
        
        # Mettre les états terminaux à l'action 0
        for terminal in self.terminals:
            if terminal < self.n_states:
                self.policy[terminal] = 0
        
        self.history = []
        
        for iteration in range(max_iterations):
            # 1. Évaluation de politique
            self.V = self.policy_evaluation(self.policy)
            
            # 2. Amélioration de politique
            new_policy, policy_stable = self.policy_improvement(self.V)
            
            # Enregistrer les statistiques
            avg_value = np.mean(self.V)
            max_value = np.max(self.V)
            
            self.history.append({
                'iteration': iteration + 1,
                'avg_value': avg_value,
                'max_value': max_value,
                'policy_stable': policy_stable
            })
            
            print(f"Itération {iteration + 1}: Valeur moyenne = {avg_value:.6f}")
            
            # Mise à jour de la politique
            self.policy = new_policy
            
            # Vérifier la convergence
            if policy_stable:
                print(f"Convergence atteinte après {iteration + 1} itérations")
                break
        
        return self.policy, self.V
    
    def train(self, max_iterations: int = 1000) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Policy Iteration.
        
        Args:
            max_iterations: Nombre maximum d'itérations
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        policy, V = self.policy_iteration(max_iterations)
        
        return {
            'policy': policy,
            'V': V,
            'history': self.history,
            'iterations': len(self.history),
            'converged': len(self.history) < max_iterations
        }
    
    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Évalue la politique apprise.
        
        Args:
            num_episodes: Nombre d'épisodes d'évaluation
            
        Returns:
            Dictionnaire avec les métriques d'évaluation
        """
        if self.policy is None:
            raise ValueError("Aucune politique disponible. Entraînez d'abord l'algorithme.")
        
        total_rewards = []
        total_steps = []
        successes = 0
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 1000:
                action = self.policy[state]
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                steps += 1
            
            total_rewards.append(episode_reward)
            total_steps.append(steps)
            
            if episode_reward > 0:
                successes += 1
        
        return {
            'average_reward': float(np.mean(total_rewards)),
            'std_reward': float(np.std(total_rewards)),
            'success_rate': successes / num_episodes,
            'average_steps': float(np.mean(total_steps)),
            'std_steps': float(np.std(total_steps))
        }
    
    def get_action(self, state: int) -> int:
        """
        Retourne l'action selon la politique apprise.
        
        Args:
            state: État courant
            
        Returns:
            Action à prendre
        """
        if self.policy is None:
            raise ValueError("Aucune politique disponible. Entraînez d'abord l'algorithme.")
        
        return self.policy[state]
    
    def save(self, filepath: str) -> None:
        """
        Sauvegarde le modèle.
        
        Args:
            filepath: Chemin de sauvegarde
        """
        model_data = {
            'policy': self.policy,
            'V': self.V,
            'history': self.history,
            'gamma': self.gamma,
            'theta': self.theta
        }
        save_model(model_data, filepath)
        
    def load(self, filepath: str) -> None:
        """
        Charge le modèle.
        
        Args:
            filepath: Chemin du modèle
        """
        model_data = load_model(filepath)
        self.policy = model_data['policy']
        self.V = model_data['V']
        self.history = model_data['history']
        self.gamma = model_data['gamma']
        self.theta = model_data['theta']


class ValueIteration:
    """
    Implémentation de l'algorithme Value Iteration.
    
    Value Iteration met à jour directement la fonction de valeur
    jusqu'à convergence, puis extrait la politique optimale.
    """
    
    def __init__(self, env: Any, gamma: float = 0.9, theta: float = 1e-6):
        """
        Initialise Value Iteration.
        
        Args:
            env: Environnement MDP
            gamma: Facteur de discount
            theta: Seuil de convergence
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = None
        self.policy = None
        self.history = []
        
        # Initialiser les structures MDP
        self._initialize_mdp_structures()
        
    def _initialize_mdp_structures(self):
        """Initialise les structures MDP nécessaires."""
        # Obtenir les informations MDP de l'environnement
        if hasattr(self.env, 'get_mdp_info'):
            # Environnement personnalisé avec get_mdp_info()
            mdp_info = self.env.get_mdp_info()
            self.states = list(mdp_info['states'])
            self.actions = mdp_info['actions']
            self.transition_matrix = mdp_info['transition_matrix']
            self.reward_matrix = mdp_info['reward_matrix']
            self.terminals = mdp_info['terminals']
            self.n_states = len(self.states)
            self.n_actions = len(self.actions)
            
        elif hasattr(self.env, 'nS') and hasattr(self.env, 'nA'):
            # Environnement gym discret
            self.n_states = self.env.nS
            self.n_actions = self.env.nA
            self.states = list(range(self.n_states))
            self.actions = list(range(self.n_actions))
            self.terminals = []
            
            # Extraire les matrices de transition et de récompense
            if hasattr(self.env, 'P'):
                self.transition_matrix, self.reward_matrix = self._extract_gym_dynamics()
            else:
                raise ValueError("Environnement gym sans dynamique disponible")
                
        else:
            # Environnement générique
            self.n_states = getattr(self.env, 'observation_space', type('obj', (object,), {'n': 16})).n
            self.n_actions = getattr(self.env, 'action_space', type('obj', (object,), {'n': 4})).n
            self.states = list(range(self.n_states))
            self.actions = list(range(self.n_actions))
            self.terminals = []
            
            # Créer des matrices par défaut
            self.transition_matrix = np.zeros((self.n_states, self.n_actions, self.n_states))
            self.reward_matrix = np.zeros((self.n_states, self.n_actions))
            
    def _extract_gym_dynamics(self):
        """Extrait les dynamiques d'un environnement Gym."""
        # Matrice de transition P(s'|s,a)
        transition_matrix = np.zeros((self.n_states, self.n_actions, self.n_states))
        # Matrice de récompense R(s,a)
        reward_matrix = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if s in self.env.P and a in self.env.P[s]:
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        transition_matrix[s, a, next_state] += prob
                        reward_matrix[s, a] += prob * reward
                        
                        if done and next_state not in self.terminals:
                            self.terminals.append(next_state)
        
        return transition_matrix, reward_matrix
        
    def value_iteration(self, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exécute l'algorithme Value Iteration complet.
        
        Args:
            max_iterations: Nombre maximum d'itérations
            
        Returns:
            Tuple (politique optimale, fonction de valeur optimale)
        """
        # Initialiser la fonction de valeur
        self.V = np.zeros(self.n_states)
        self.history = []
        
        for iteration in range(max_iterations):
            delta = 0.0
            
            for s in self.states:
                if s in self.terminals:
                    continue
                    
                v = self.V[s]
                best_value = -np.inf
                
                # Trouver la meilleure action
                for a in self.actions:
                    value = self.reward_matrix[s, a]
                    for s_prime in self.states:
                        value += self.gamma * self.transition_matrix[s, a, s_prime] * self.V[s_prime]
                    
                    if value > best_value:
                        best_value = value
                
                self.V[s] = best_value
                delta = max(delta, abs(v - self.V[s]))
            
            # Enregistrer les statistiques
            avg_value = np.mean(self.V)
            max_value = np.max(self.V)
            
            self.history.append({
                'iteration': iteration + 1,
                'avg_value': avg_value,
                'max_value': max_value,
                'delta': delta
            })
            
            print(f"Itération {iteration + 1}: Valeur moyenne = {avg_value:.6f}, Delta = {delta:.6f}")
            
            # Vérifier la convergence
            if delta < self.theta:
                print(f"Convergence atteinte après {iteration + 1} itérations")
                break
        
        # Extraire la politique optimale
        self.policy = self._extract_policy(self.V)
        
        return self.policy, self.V
    
    def _extract_policy(self, V: np.ndarray) -> np.ndarray:
        """
        Extrait la politique optimale de la fonction de valeur.
        
        Args:
            V: Fonction de valeur
            
        Returns:
            Politique optimale
        """
        policy = np.zeros(self.n_states, dtype=int)
        
        for s in self.states:
            if s in self.terminals:
                policy[s] = 0
                continue
                
            best_action = None
            best_value = -np.inf
            
            for a in self.actions:
                value = self.reward_matrix[s, a]
                for s_prime in self.states:
                    value += self.gamma * self.transition_matrix[s, a, s_prime] * V[s_prime]
                
                if value > best_value:
                    best_value = value
                    best_action = a
            
            policy[s] = best_action
        
        return policy
    
    def train(self, max_iterations: int = 1000) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Value Iteration.
        
        Args:
            max_iterations: Nombre maximum d'itérations
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        policy, V = self.value_iteration(max_iterations)
        
        return {
            'policy': policy,
            'V': V,
            'history': self.history,
            'iterations': len(self.history),
            'converged': len(self.history) < max_iterations
        }
    
    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Évalue la politique apprise.
        
        Args:
            num_episodes: Nombre d'épisodes d'évaluation
            
        Returns:
            Dictionnaire avec les métriques d'évaluation
        """
        if self.policy is None:
            raise ValueError("Aucune politique disponible. Entraînez d'abord l'algorithme.")
        
        total_rewards = []
        total_steps = []
        successes = 0
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 1000:
                action = self.policy[state]
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                steps += 1
            
            total_rewards.append(episode_reward)
            total_steps.append(steps)
            
            if episode_reward > 0:
                successes += 1
        
        return {
            'average_reward': float(np.mean(total_rewards)),
            'std_reward': float(np.std(total_rewards)),
            'success_rate': successes / num_episodes,
            'average_steps': float(np.mean(total_steps)),
            'std_steps': float(np.std(total_steps))
        }
    
    def get_action(self, state: int) -> int:
        """
        Retourne l'action selon la politique apprise.
        
        Args:
            state: État courant
            
        Returns:
            Action à prendre
        """
        if self.policy is None:
            raise ValueError("Aucune politique disponible. Entraînez d'abord l'algorithme.")
        
        return self.policy[state]
    
    def save(self, filepath: str) -> None:
        """
        Sauvegarde le modèle.
        
        Args:
            filepath: Chemin de sauvegarde
        """
        model_data = {
            'policy': self.policy,
            'V': self.V,
            'history': self.history,
            'gamma': self.gamma,
            'theta': self.theta
        }
        save_model(model_data, filepath)
        
    def load(self, filepath: str) -> None:
        """
        Charge le modèle.
        
        Args:
            filepath: Chemin du modèle
        """
        model_data = load_model(filepath)
        self.policy = model_data['policy']
        self.V = model_data['V']
        self.history = model_data['history']
        self.gamma = model_data['gamma']
        self.theta = model_data['theta']