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
import time


# ALGORITHME POLICY ITERATION
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
        
    def policy_evaluation(self, policy: np.ndarray) -> np.ndarray:
        """
        Évalue une politique donnée.
        
        Args:
            policy: Politique à évaluer (actions pour chaque état)
            
        Returns:
            Fonction de valeur d'état
        """
        V = np.copy(self.V) if self.V is not None else np.random.random((self.n_states,))
        
        # Les états terminaux ont une valeur de 0
        if hasattr(self, 'terminal_states'):
            V[self.terminal_states] = 0.0
        
        while True:
            delta = 0.0
            
            for s in self.states:
                v = V[s]
                total = 0.0
                
                # Calculer la valeur pour l'action dictée par la politique
                action = policy[s]
                
                if hasattr(self, 'reward_matrix') and self.reward_matrix is not None:
                    # Nouveau format avec reward_matrix (même logique que ValueIteration)
                    for s_p in self.states:
                        prob = self.transition_probs[s, action, s_p]
                        reward = self.reward_matrix[s, action]
                        total += prob * (reward + self.gamma * V[s_p])
                else:
                    # Ancien format avec récompenses multiples
                    for s_p in self.states:
                        for r_index in range(len(self.rewards)):
                            r = self.rewards[r_index]
                            prob = self.transition_probs[s, action, s_p, r_index]
                            total += prob * (r + self.gamma * V[s_p])
                
                V[s] = total
                abs_diff = np.abs(v - V[s])
                delta = np.maximum(delta, abs_diff)
            
            if delta < self.theta:
                break
        
        return V
        
    def policy_improvement(self, V: np.ndarray) -> np.ndarray:
        """
        Améliore la politique basée sur la fonction de valeur.
        
        Args:
            V: Fonction de valeur d'état
            
        Returns:
            Politique améliorée
        """
        new_policy = np.copy(self.policy)
        policy_stable = True
        
        for s in self.states:
            old_action = new_policy[s]
            best_action = None
            best_score = -np.inf
            
            # Évaluer toutes les actions possibles
            for a in self.actions:
                score = 0.0
                
                if hasattr(self, 'reward_matrix') and self.reward_matrix is not None:
                    # Nouveau format avec reward_matrix (même logique que ValueIteration)
                    for s_p in self.states:
                        prob = self.transition_probs[s, a, s_p]
                        reward = self.reward_matrix[s, a]
                        score += prob * (reward + self.gamma * V[s_p])
                else:
                    # Ancien format avec récompenses multiples
                    for s_p in self.states:
                        for r_index in range(len(self.rewards)):
                            r = self.rewards[r_index]
                            prob = self.transition_probs[s, a, s_p, r_index]
                            score += prob * (r + self.gamma * V[s_p])
                
                if best_action is None or score > best_score:
                    best_action = a
                    best_score = score
            
            if best_action != old_action:
                policy_stable = False
            
            new_policy[s] = best_action
        
        return new_policy, policy_stable
        
    def train(self, max_iterations: int = 1000) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Policy Iteration.
        
        Args:
            max_iterations: Nombre maximum d'itérations
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        # Initialiser les structures nécessaires
        self._initialize_mdp_structures()
        
        # Initialiser V et politique
        self.V = np.random.random((self.n_states,))
        if hasattr(self, 'terminal_states'):
            self.V[self.terminal_states] = 0.0
        
        # Politique initiale aléatoire
        self.policy = np.array([np.random.choice(self.actions) for _ in self.states])
        if hasattr(self, 'terminal_states'):
            self.policy[self.terminal_states] = 0
        
        # Historique d'entraînement
        self.history = []
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            # 1. Policy Evaluation
            self.V = self.policy_evaluation(self.policy)
            
            # 2. Policy Improvement
            new_policy, policy_stable = self.policy_improvement(self.V)
            
            # Enregistrer les statistiques
            avg_value = np.mean(self.V)
            max_value = np.max(self.V)
            
            self.history.append({
                'iteration': iteration,
                'avg_value': avg_value,
                'max_value': max_value,
                'policy_stable': policy_stable
            })
            
            # Mise à jour de la politique
            self.policy = new_policy
            
            # Vérifier la convergence
            if policy_stable:
                print(f"Convergence atteinte après {iteration} itérations")
                break
        
        return {
            'policy': self.policy,
            'V': self.V,
            'history': self.history,
            'iterations': iteration,
            'converged': policy_stable if 'policy_stable' in locals() else False
        }
    
    def evaluate(self, num_episodes: int = 100) -> dict:
        """
        Évalue la politique trouvée sur l'environnement.
        Retourne récompense moyenne, min, max, nb itérations, temps d'exécution, courbe d'apprentissage.
        """
        rewards = []
        steps_list = []
        start = time.time()
        for _ in range(num_episodes):
            state = self.env.reset()
            done, G, steps = False, 0.0, 0
            while not done and steps < 1000:
                a = self.policy[state]
                state, r, done, _ = self.env.step(a)
                G += r
                steps += 1
            rewards.append(G)
            steps_list.append(steps)
        elapsed = time.time() - start
        # Courbe d'apprentissage = historique des valeurs moyennes par itération
        learning_curve = [h['avg_value'] for h in self.history] if self.history else []
        return {
            'avg_reward': float(np.mean(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'iterations': len(self.history),
            'converged': self.history[-1]['policy_stable'] if self.history else None,
            'execution_time': elapsed,
            'learning_curve': learning_curve,
            'rewards': rewards,
            'steps_per_episode': steps_list
        }
    
    def _initialize_mdp_structures(self):
        """
        Initialise les structures nécessaires pour un MDP.
        """
        if hasattr(self.env, 'get_mdp_info'):
            # Utiliser les informations fournies par l'environnement
            mdp_info = self.env.get_mdp_info()
            self.states = list(mdp_info['states'])
            self.actions = mdp_info['actions']
            self.n_states = len(self.states)
            self.n_actions = len(self.actions)
            self.terminal_states = mdp_info['terminals']
            self.transition_probs = mdp_info['transition_matrix']
            self.reward_matrix = mdp_info['reward_matrix']
            self.rewards = [0.0, 1.0]  # Garde pour compatibilité avec l'ancien format
            
        elif hasattr(self.env, 'nS') and hasattr(self.env, 'nA'):
            # Environnement gym discret (comme FrozenLake)
            self.n_states = self.env.nS
            self.n_actions = self.env.nA
            self.states = list(range(self.n_states))
            self.actions = list(range(self.n_actions))
            
            # Initialiser les récompenses possibles
            self.rewards = [0.0, 1.0]  # À adapter selon votre environnement
            
            # Initialiser les probabilités de transition
            self._initialize_transition_probabilities()
            
        else:
            # Environnement générique - à adapter
            self.n_states = getattr(self.env, 'observation_space', type('obj', (object,), {'n': 16})).n
            self.n_actions = getattr(self.env, 'action_space', type('obj', (object,), {'n': 4})).n
            self.states = list(range(self.n_states))
            self.actions = list(range(self.n_actions))
            self.rewards = [0.0, 1.0]
            
            # Probabilités de transition par défaut (à adapter)
            self.transition_probs = np.ones((self.n_states, self.n_actions, self.n_states, len(self.rewards))) / (self.n_states * len(self.rewards))
    
    def _initialize_transition_probabilities(self):
        """
        Initialise les probabilités de transition depuis l'environnement.
        """
        self.transition_probs = np.zeros((self.n_states, self.n_actions, self.n_states, len(self.rewards)))
        
        # Si l'environnement a des probabilités de transition accessibles
        if hasattr(self.env, 'P'):
            for s in self.states:
                for a in self.actions:
                    if s in self.env.P and a in self.env.P[s]:
                        for prob, next_state, reward, done in self.env.P[s][a]:
                            # Trouver l'index de la récompense
                            r_index = 0
                            if reward in self.rewards:
                                r_index = self.rewards.index(reward)
                            else:
                                # Ajouter la nouvelle récompense
                                self.rewards.append(reward)
                                r_index = len(self.rewards) - 1
                                # Redimensionner le tableau des probabilités
                                new_shape = (self.n_states, self.n_actions, self.n_states, len(self.rewards))
                                new_probs = np.zeros(new_shape)
                                new_probs[:, :, :, :-1] = self.transition_probs
                                self.transition_probs = new_probs
                            
                            self.transition_probs[s, a, next_state, r_index] = prob
        else:
            # Distribution uniforme par défaut
            uniform_prob = 1.0 / (self.n_states * len(self.rewards))
            self.transition_probs.fill(uniform_prob)
        
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        model_data = {
            'V': self.V,
            'policy': self.policy,
            'gamma': self.gamma,
            'theta': self.theta,
            'history': self.history
        }
        save_model(model_data, filepath)
        
    def load(self, filepath: str):
        """Charge un modèle sauvegardé."""
        model_data = load_model(filepath)
        self.V = model_data['V']
        self.policy = model_data['policy']
        self.gamma = model_data['gamma']
        self.theta = model_data['theta']
        self.history = model_data['history']



# ALGORITHME VALUE ITERATION
class ValueIteration:
    """
    Implémentation de l'algorithme Value Iteration.
    
    Value Iteration met à jour directement la fonction de valeur en utilisant
    l'équation de Bellman jusqu'à convergence vers la fonction de valeur optimale.
    """
    
    def __init__(self, env: Any, gamma: float = 0.999999, theta: float = 1e-5):
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
        
    def value_update(self, V: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Met à jour la fonction de valeur selon l'équation de Bellman.
        
        Args:
            V: Fonction de valeur actuelle
            
        Returns:
            Tuple contenant la fonction de valeur mise à jour et le delta maximal
        """
        new_V = np.copy(V)
        delta = 0.0
        
        for s in self.states:
            if s in self.terminal_states:
                continue  # Ne pas mettre à jour les états terminaux
            
            v = V[s]  # Ancienne valeur
            max_value = float('-inf')
            
            # Maximiser sur toutes les actions possibles
            for a in self.actions:
                value = 0.0
                
                if hasattr(self, 'reward_matrix') and self.reward_matrix is not None:
                    # Utiliser les matrices de l'environnement directement
                    for s_p in self.states:
                        prob = self.transition_probs[s, a, s_p]
                        reward = self.reward_matrix[s, a]
                        value += prob * (reward + self.gamma * V[s_p])
                else:
                    # Ancien format avec récompenses multiples
                    for s_p in self.states:
                        for r_idx, r in enumerate(self.rewards):
                            prob = self.transition_probs[s, a, s_p, r_idx]
                            value += prob * (r + self.gamma * V[s_p])
                            
                max_value = max(max_value, value)
            
            new_V[s] = max_value
            delta = max(delta, abs(v - new_V[s]))
        
        return new_V, delta
        
    def extract_policy(self, V: np.ndarray) -> np.ndarray:
        """
        Extrait la politique optimale de la fonction de valeur.
        
        Args:
            V: Fonction de valeur optimale
            
        Returns:
            Politique optimale (matrice de probabilités)
        """
        # Politique sous forme de matrice de probabilités
        policy = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        
        for s in self.states:
            if s in self.terminal_states:
                continue
            
            action_values = []
            for a in self.actions:
                value = 0.0
                
                if hasattr(self, 'reward_matrix') and self.reward_matrix is not None:
                    # Utiliser les matrices de l'environnement directement
                    for s_p in self.states:
                        prob = self.transition_probs[s, a, s_p]
                        reward = self.reward_matrix[s, a]
                        value += prob * (reward + self.gamma * V[s_p])
                else:
                    # Ancien format avec récompenses multiples
                    for s_p in self.states:
                        for r_idx, r in enumerate(self.rewards):
                            prob = self.transition_probs[s, a, s_p, r_idx]
                            value += prob * (r + self.gamma * V[s_p])
                            
                action_values.append(value)
            
            # Politique déterministe optimale
            best_action = np.argmax(action_values)
            policy[s, best_action] = 1.0
        
        return policy
        
    def train(self, max_iterations: int = 1000) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Value Iteration.
        
        Args:
            max_iterations: Nombre maximum d'itérations
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        # Initialiser les structures MDP
        self._initialize_mdp_structures()
        
        # Étape 1: Initialiser les valeurs à zéro
        self.V = np.zeros(self.n_states, dtype=np.float64)
        
        # Historique d'entraînement
        self.history = []
        
        iteration = 0
        converged = False
        
        while iteration < max_iterations:
            iteration += 1
            
            # Mise à jour de la fonction de valeur
            new_V, delta = self.value_update(self.V)
            
            # Enregistrer les statistiques
            avg_value = np.mean(self.V)
            max_value = np.max(self.V)
            
            self.history.append({
                'iteration': iteration,
                'delta': delta,
                'avg_value': avg_value,
                'max_value': max_value,
                'converged': delta < self.theta
            })
            
            # Mise à jour de V
            self.V = new_V
            
            # Vérifier la convergence
            if delta < self.theta:
                converged = True
                print(f"Convergence atteinte après {iteration} itérations (delta = {delta:.6f})")
                break
        
        # Étape 3: Extraire la politique optimale
        self.policy = self.extract_policy(self.V)
        
        return {
            'V': self.V,
            'policy': self.policy,
            'history': self.history,
            'iterations': iteration,
            'converged': converged,
            'final_delta': delta if 'delta' in locals() else None
        }
    
    def evaluate(self, num_episodes: int = 100) -> dict:
        """
        Évalue la politique trouvée sur l'environnement.
        Retourne récompense moyenne, min, max, nb itérations, temps d'exécution, courbe d'apprentissage.
        """
        rewards = []
        steps_list = []
        start = time.time()
        # Politique déterministe extraite de self.policy (matrice one-hot)
        policy_indices = np.argmax(self.policy, axis=1) if self.policy is not None else None
        for _ in range(num_episodes):
            state = self.env.reset()
            done, G, steps = False, 0.0, 0
            while not done and steps < 1000:
                a = policy_indices[state] if policy_indices is not None else 0
                state, r, done, _ = self.env.step(a)
                G += r
                steps += 1
            rewards.append(G)
            steps_list.append(steps)
        elapsed = time.time() - start
        learning_curve = [h['avg_value'] for h in self.history] if self.history else []
        return {
            'avg_reward': float(np.mean(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'iterations': len(self.history),
            'converged': self.history[-1]['converged'] if self.history else None,
            'execution_time': elapsed,
            'learning_curve': learning_curve,
            'rewards': rewards,
            'steps_per_episode': steps_list
        }
    
    def _initialize_mdp_structures(self):
        """
        Initialise les structures nécessaires pour un MDP.
        """
        if hasattr(self.env, 'get_mdp_info'):
            # Utiliser les informations fournies par l'environnement
            mdp_info = self.env.get_mdp_info()
            self.states = list(mdp_info['states'])
            self.actions = mdp_info['actions']
            self.n_states = len(self.states)
            self.n_actions = len(self.actions)
            self.terminal_states = mdp_info['terminals']
            self.transition_probs = mdp_info['transition_matrix']
            self.reward_matrix = mdp_info['reward_matrix']
            self.rewards = [0.0, 1.0]  # Garde pour compatibilité
            
        elif hasattr(self.env, 'nS') and hasattr(self.env, 'nA'):
            # Environnement gym discret
            self.n_states = self.env.nS
            self.n_actions = self.env.nA
            self.states = list(range(self.n_states))
            self.actions = list(range(self.n_actions))
            
            # Initialiser les récompenses possibles
            self.rewards = [0.0, 1.0]  # À adapter selon l'environnement
            self.reward_matrix = None
            
            # Identifier les états terminaux
            self.terminal_states = []
            if hasattr(self.env, 'desc'):
                # Pour FrozenLake par exemple
                desc = self.env.desc
                for i in range(desc.shape[0]):
                    for j in range(desc.shape[1]):
                        if desc[i, j] in [b'H', b'G']:  # Hole ou Goal
                            self.terminal_states.append(i * desc.shape[1] + j)
            
            # Initialiser les probabilités de transition
            self._initialize_transition_probabilities()
            
        else:
            # Environnement générique
            self.n_states = getattr(self.env, 'observation_space', type('obj', (object,), {'n': 16})).n
            self.n_actions = getattr(self.env, 'action_space', type('obj', (object,), {'n': 4})).n
            self.states = list(range(self.n_states))
            self.actions = list(range(self.n_actions))
            self.rewards = [0.0, 1.0]
            self.reward_matrix = None
            self.terminal_states = []
            
            # Probabilités de transition par défaut
            self.transition_probs = np.ones((self.n_states, self.n_actions, self.n_states, len(self.rewards))) / (self.n_states * len(self.rewards))
    
    def _initialize_transition_probabilities(self):
        """
        Initialise les probabilités de transition depuis l'environnement.
        """
        self.transition_probs = np.zeros((self.n_states, self.n_actions, self.n_states, len(self.rewards)))
        
        # Si l'environnement a des probabilités de transition accessibles
        if hasattr(self.env, 'P'):
            for s in self.states:
                for a in self.actions:
                    if s in self.env.P and a in self.env.P[s]:
                        for prob, next_state, reward, done in self.env.P[s][a]:
                            # Trouver l'index de la récompense
                            r_index = 0
                            if reward in self.rewards:
                                r_index = self.rewards.index(reward)
                            else:
                                # Ajouter la nouvelle récompense
                                self.rewards.append(reward)
                                r_index = len(self.rewards) - 1
                                # Redimensionner le tableau des probabilités
                                new_shape = (self.n_states, self.n_actions, self.n_states, len(self.rewards))
                                new_probs = np.zeros(new_shape)
                                new_probs[:, :, :, :-1] = self.transition_probs
                                self.transition_probs = new_probs
                            
                            self.transition_probs[s, a, next_state, r_index] = prob
        else:
            # Distribution uniforme par défaut
            uniform_prob = 1.0 / (self.n_states * len(self.rewards))
            self.transition_probs.fill(uniform_prob)
        
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        model_data = {
            'V': self.V,
            'policy': self.policy,
            'gamma': self.gamma,
            'theta': self.theta,
            'history': self.history
        }
        save_model(model_data, filepath)
        
    def load(self, filepath: str):
        """Charge un modèle sauvegardé."""
        model_data = load_model(filepath)
        self.V = model_data['V']
        self.policy = model_data['policy']
        self.gamma = model_data['gamma']
        self.theta = model_data['theta']
        self.history = model_data['history']
