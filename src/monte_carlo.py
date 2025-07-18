"""
Module de Monte Carlo pour le Reinforcement Learning.

Ce module implémente les algorithmes de Monte Carlo :
- Monte Carlo Exploring Starts (MC-ES)
- On-policy First-Visit Monte Carlo
- Off-policy Monte Carlo avec Importance Sampling

Ces algorithmes apprennent à partir d'épisodes complets sans connaître
la dynamique de l'environnement.
"""

import numpy as np
import random
from typing import Dict, Tuple, Optional, Any, List
from collections import defaultdict
from src.utils_io import save_model, load_model


class MonteCarloES:
    """
    Implémentation de Monte Carlo Exploring Starts.
    
    MC-ES assure que tous les couples état-action ont une probabilité
    non nulle d'être sélectionnés comme point de départ.
    """
    
    def __init__(self, env: Any, gamma: float = 1.0):
        """
        Initialise Monte Carlo ES.
        
        Args:
            env: Environnement
            gamma: Facteur de discount
        """
        self.env = env
        self.gamma = gamma
        self.Q = None
        self.policy = None
        self.returns_sum = {}
        self.returns_count = {}
        self.history = []
        
        # Initialiser les structures MDP
        self._initialize_mdp_structures()
        
    def _initialize_mdp_structures(self):
        """Initialise les structures MDP nécessaires."""
        # Déterminer le nombre d'états et d'actions
        if hasattr(self.env, 'nS') and hasattr(self.env, 'nA'):
            # Environnement gym discret
            self.nS = self.env.nS
            self.nA = self.env.nA
        elif hasattr(self.env, 'observation_space') and hasattr(self.env, 'action_space'):
            # Environnement gym avec observation/action spaces
            self.nS = getattr(self.env.observation_space, 'n', 16)
            self.nA = getattr(self.env.action_space, 'n', 4)
        else:
            # Environnement personnalisé - essayer d'obtenir les infos MDP
            if hasattr(self.env, 'get_mdp_info'):
                mdp_info = self.env.get_mdp_info()
                self.nS = len(list(mdp_info['states']))
                self.nA = len(mdp_info['actions'])
            else:
                # Valeurs par défaut
                self.nS = 16
                self.nA = 4
        
        # Initialiser Q et politique
        self.Q = np.zeros((self.nS, self.nA), dtype=float)
        self.policy = np.zeros(self.nS, dtype=int)
        
        # Initialiser les dictionnaires de retours
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(int)
        
    def generate_episode(self, start_state: int, start_action: int) -> List[Tuple[int, int, float]]:
        """
        Génère un épisode en commençant par l'état et l'action donnés.
        
        Args:
            start_state: État de départ
            start_action: Action de départ
            
        Returns:
            Liste des transitions (état, action, récompense)
        """
        episode = []
        state = start_state
        action = start_action
        
        # Réinitialiser l'environnement et forcer l'état de départ
        self.env.reset()
        if hasattr(self.env, 'state'):
            self.env.state = start_state
        elif hasattr(self.env, 'current_state'):
            self.env.current_state = start_state
        
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            # Prendre l'action
            try:
                next_state, reward, done, _ = self.env.step(action)
                episode.append((state, action, reward))
                
                if done:
                    break
                
                state = next_state
                
                # Choisir la prochaine action selon la politique
                action = self.policy[state]
                
            except Exception as e:
                # Si l'environnement ne supporte pas cette interaction
                break
            
            steps += 1
        
        return episode
    
    def update_q_function(self, episode: List[Tuple[int, int, float]]) -> None:
        """
        Met à jour la fonction Q basée sur un épisode.
        
        Args:
            episode: Liste des transitions (état, action, récompense)
        """
        # Calculer les retours
        returns = 0.0
        visited_pairs = set()
        
        # Parcourir l'épisode en sens inverse
        for i in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[i]
            returns = self.gamma * returns + reward
            
            # First-visit Monte Carlo
            if (state, action) not in visited_pairs:
                visited_pairs.add((state, action))
                
                # Mettre à jour les moyennes
                self.returns_count[(state, action)] += 1
                self.returns_sum[(state, action)] += returns
                
                # Mettre à jour Q
                self.Q[state, action] = self.returns_sum[(state, action)] / self.returns_count[(state, action)]
    
    def improve_policy(self) -> None:
        """
        Améliore la politique basée sur la fonction Q actuelle.
        """
        for state in range(self.nS):
            # Choisir l'action avec la plus haute valeur Q
            self.policy[state] = np.argmax(self.Q[state])
    
    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Monte Carlo ES.
        
        Args:
            num_episodes: Nombre d'épisodes d'entraînement
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        # Initialiser la politique aléatoirement
        for state in range(self.nS):
            self.policy[state] = random.randint(0, self.nA - 1)
        
        self.history = []
        
        for episode_num in range(num_episodes):
            # Exploring starts: choisir un état et une action aléatoires
            start_state = random.randint(0, self.nS - 1)
            start_action = random.randint(0, self.nA - 1)
            
            # Générer un épisode
            episode = self.generate_episode(start_state, start_action)
            
            if episode:  # Si l'épisode n'est pas vide
                # Mettre à jour la fonction Q
                self.update_q_function(episode)
                
                # Améliorer la politique
                self.improve_policy()
                
                # Enregistrer les statistiques
                episode_reward = sum(reward for _, _, reward in episode)
                avg_q = np.mean(self.Q)
                
                self.history.append({
                    'episode': episode_num + 1,
                    'reward': episode_reward,
                    'avg_q': avg_q,
                    'episode_length': len(episode)
                })
                
                if (episode_num + 1) % 100 == 0:
                    print(f"Épisode {episode_num + 1}: Récompense = {episode_reward:.3f}, Q moyen = {avg_q:.6f}")
        
        return {
            'Q': self.Q,
            'policy': self.policy,
            'history': self.history,
            'returns_sum': dict(self.returns_sum),
            'returns_count': dict(self.returns_count)
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
            'average_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'success_rate': successes / num_episodes,
            'average_steps': np.mean(total_steps),
            'std_steps': np.std(total_steps)
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
            'Q': self.Q,
            'policy': self.policy,
            'returns_sum': dict(self.returns_sum),
            'returns_count': dict(self.returns_count),
            'history': self.history,
            'gamma': self.gamma
        }
        save_model(model_data, filepath)
        
    def load(self, filepath: str) -> None:
        """
        Charge le modèle.
        
        Args:
            filepath: Chemin du modèle
        """
        model_data = load_model(filepath)
        self.Q = model_data['Q']
        self.policy = model_data['policy']
        self.returns_sum = defaultdict(float, model_data['returns_sum'])
        self.returns_count = defaultdict(int, model_data['returns_count'])
        self.history = model_data['history']
        self.gamma = model_data['gamma']


class OnPolicyMC:
    """
    Implémentation de On-policy First-Visit Monte Carlo.
    
    Cet algorithme utilise une politique epsilon-greedy pour explorer
    et met à jour la fonction Q et la politique simultanément.
    """
    
    def __init__(self, env: Any, gamma: float = 1.0, epsilon: float = 0.1):
        """
        Initialise On-policy Monte Carlo.
        
        Args:
            env: Environnement
            gamma: Facteur de discount
            epsilon: Paramètre epsilon-greedy
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.Q = None
        self.policy = None
        self.returns_sum = {}
        self.returns_count = {}
        self.history = []
        
        # Initialiser les structures MDP
        self._initialize_mdp_structures()
        
    def _initialize_mdp_structures(self):
        """Initialise les structures MDP nécessaires."""
        # Déterminer le nombre d'états et d'actions
        if hasattr(self.env, 'nS') and hasattr(self.env, 'nA'):
            # Environnement gym discret
            self.nS = self.env.nS
            self.nA = self.env.nA
        elif hasattr(self.env, 'observation_space') and hasattr(self.env, 'action_space'):
            # Environnement gym avec observation/action spaces
            self.nS = getattr(self.env.observation_space, 'n', 16)
            self.nA = getattr(self.env.action_space, 'n', 4)
        else:
            # Environnement personnalisé - essayer d'obtenir les infos MDP
            if hasattr(self.env, 'get_mdp_info'):
                mdp_info = self.env.get_mdp_info()
                self.nS = len(list(mdp_info['states']))
                self.nA = len(mdp_info['actions'])
            else:
                # Valeurs par défaut
                self.nS = 16
                self.nA = 4
        
        # Initialiser Q et politique
        self.Q = np.zeros((self.nS, self.nA), dtype=float)
        self.policy = np.zeros(self.nS, dtype=int)
        
        # Initialiser les dictionnaires de retours
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(int)
    
    def epsilon_greedy_action(self, state: int) -> int:
        """
        Choisit une action selon la politique epsilon-greedy.
        
        Args:
            state: État courant
            
        Returns:
            Action choisie
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.nA - 1)
        else:
            return np.argmax(self.Q[state])
    
    def generate_episode(self) -> List[Tuple[int, int, float]]:
        """
        Génère un épisode en suivant la politique epsilon-greedy.
        
        Returns:
            Liste des transitions (état, action, récompense)
        """
        episode = []
        state = self.env.reset()
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            # Choisir une action
            action = self.epsilon_greedy_action(state)
            
            # Prendre l'action
            try:
                next_state, reward, done, _ = self.env.step(action)
                episode.append((state, action, reward))
                state = next_state
                
            except Exception as e:
                break
            
            steps += 1
        
        return episode
    
    def update_q_function(self, episode: List[Tuple[int, int, float]]) -> None:
        """
        Met à jour la fonction Q basée sur un épisode.
        
        Args:
            episode: Liste des transitions (état, action, récompense)
        """
        # Calculer les retours
        returns = 0.0
        visited_pairs = set()
        
        # Parcourir l'épisode en sens inverse
        for i in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[i]
            returns = self.gamma * returns + reward
            
            # First-visit Monte Carlo
            if (state, action) not in visited_pairs:
                visited_pairs.add((state, action))
                
                # Mettre à jour les moyennes
                self.returns_count[(state, action)] += 1
                self.returns_sum[(state, action)] += returns
                
                # Mettre à jour Q
                self.Q[state, action] = self.returns_sum[(state, action)] / self.returns_count[(state, action)]
    
    def improve_policy(self) -> None:
        """
        Améliore la politique basée sur la fonction Q actuelle.
        """
        for state in range(self.nS):
            # Choisir l'action avec la plus haute valeur Q
            self.policy[state] = np.argmax(self.Q[state])
    
    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Entraîne l'algorithme On-policy Monte Carlo.
        
        Args:
            num_episodes: Nombre d'épisodes d'entraînement
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        self.history = []
        
        for episode_num in range(num_episodes):
            # Générer un épisode
            episode = self.generate_episode()
            
            if episode:  # Si l'épisode n'est pas vide
                # Mettre à jour la fonction Q
                self.update_q_function(episode)
                
                # Améliorer la politique
                self.improve_policy()
                
                # Decay epsilon
                self.epsilon = max(0.01, self.epsilon * 0.995)
                
                # Enregistrer les statistiques
                episode_reward = sum(reward for _, _, reward in episode)
                avg_q = np.mean(self.Q)
                
                self.history.append({
                    'episode': episode_num + 1,
                    'reward': episode_reward,
                    'avg_q': avg_q,
                    'epsilon': self.epsilon,
                    'episode_length': len(episode)
                })
                
                if (episode_num + 1) % 100 == 0:
                    print(f"Épisode {episode_num + 1}: Récompense = {episode_reward:.3f}, Epsilon = {self.epsilon:.3f}")
        
        return {
            'Q': self.Q,
            'policy': self.policy,
            'history': self.history,
            'returns_sum': dict(self.returns_sum),
            'returns_count': dict(self.returns_count)
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
            'average_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'success_rate': successes / num_episodes,
            'average_steps': np.mean(total_steps),
            'std_steps': np.std(total_steps)
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
            'Q': self.Q,
            'policy': self.policy,
            'returns_sum': dict(self.returns_sum),
            'returns_count': dict(self.returns_count),
            'history': self.history,
            'gamma': self.gamma,
            'epsilon': self.epsilon
        }
        save_model(model_data, filepath)
        
    def load(self, filepath: str) -> None:
        """
        Charge le modèle.
        
        Args:
            filepath: Chemin du modèle
        """
        model_data = load_model(filepath)
        self.Q = model_data['Q']
        self.policy = model_data['policy']
        self.returns_sum = defaultdict(float, model_data['returns_sum'])
        self.returns_count = defaultdict(int, model_data['returns_count'])
        self.history = model_data['history']
        self.gamma = model_data['gamma']
        self.epsilon = model_data['epsilon']


class OffPolicyMC:
    """
    Implémentation de Off-policy Monte Carlo avec Importance Sampling.
    
    Cet algorithme utilise une politique de comportement (behavior policy)
    pour explorer et une politique cible (target policy) pour évaluer.
    """
    
    def __init__(self, env: Any, gamma: float = 1.0, epsilon: float = 0.1):
        """
        Initialise Off-policy Monte Carlo.
        
        Args:
            env: Environnement
            gamma: Facteur de discount
            epsilon: Paramètre epsilon-greedy pour la politique de comportement
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = None
        self.target_policy = None
        self.C = None  # Cumulative weights
        self.history = []
        
        # Initialiser les structures MDP
        self._initialize_mdp_structures()
        
    def _initialize_mdp_structures(self):
        """Initialise les structures MDP nécessaires."""
        # Déterminer le nombre d'états et d'actions
        if hasattr(self.env, 'nS') and hasattr(self.env, 'nA'):
            # Environnement gym discret
            self.nS = self.env.nS
            self.nA = self.env.nA
        elif hasattr(self.env, 'observation_space') and hasattr(self.env, 'action_space'):
            # Environnement gym avec observation/action spaces
            self.nS = getattr(self.env.observation_space, 'n', 16)
            self.nA = getattr(self.env.action_space, 'n', 4)
        else:
            # Environnement personnalisé - essayer d'obtenir les infos MDP
            if hasattr(self.env, 'get_mdp_info'):
                mdp_info = self.env.get_mdp_info()
                self.nS = len(list(mdp_info['states']))
                self.nA = len(mdp_info['actions'])
            else:
                # Valeurs par défaut
                self.nS = 16
                self.nA = 4
        
        # Initialiser Q, politique cible et poids cumulatifs
        self.Q = np.zeros((self.nS, self.nA), dtype=float)
        self.target_policy = np.zeros(self.nS, dtype=int)
        self.C = np.zeros((self.nS, self.nA), dtype=float)
    
    def behavior_policy(self, state: int) -> int:
        """
        Politique de comportement epsilon-greedy.
        
        Args:
            state: État courant
            
        Returns:
            Action choisie
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.nA - 1)
        else:
            return np.argmax(self.Q[state])
    
    def generate_episode(self) -> List[Tuple[int, int, float]]:
        """
        Génère un épisode en suivant la politique de comportement.
        
        Returns:
            Liste des transitions (état, action, récompense)
        """
        episode = []
        state = self.env.reset()
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            # Choisir une action selon la politique de comportement
            action = self.behavior_policy(state)
            
            # Prendre l'action
            try:
                next_state, reward, done, _ = self.env.step(action)
                episode.append((state, action, reward))
                state = next_state
                
            except Exception as e:
                break
            
            steps += 1
        
        return episode
    
    def update_q_function(self, episode: List[Tuple[int, int, float]]) -> None:
        """
        Met à jour la fonction Q avec importance sampling.
        
        Args:
            episode: Liste des transitions (état, action, récompense)
        """
        G = 0.0
        W = 1.0
        
        # Parcourir l'épisode en sens inverse
        for i in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[i]
            G = self.gamma * G + reward
            
            # Mettre à jour C et Q
            self.C[state, action] += W
            self.Q[state, action] += (W / self.C[state, action]) * (G - self.Q[state, action])
            
            # Mettre à jour la politique cible (greedy)
            self.target_policy[state] = np.argmax(self.Q[state])
            
            # Si l'action n'est pas celle de la politique cible, arrêter
            if action != self.target_policy[state]:
                break
            
            # Mettre à jour le poids d'importance
            # Probabilité de la politique cible (déterministe)
            target_prob = 1.0
            
            # Probabilité de la politique de comportement (epsilon-greedy)
            if action == np.argmax(self.Q[state]):
                behavior_prob = 1.0 - self.epsilon + self.epsilon / self.nA
            else:
                behavior_prob = self.epsilon / self.nA
            
            W *= target_prob / behavior_prob
    
    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Off-policy Monte Carlo.
        
        Args:
            num_episodes: Nombre d'épisodes d'entraînement
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        self.history = []
        
        for episode_num in range(num_episodes):
            # Générer un épisode
            episode = self.generate_episode()
            
            if episode:  # Si l'épisode n'est pas vide
                # Mettre à jour la fonction Q
                self.update_q_function(episode)
                
                # Enregistrer les statistiques
                episode_reward = sum(reward for _, _, reward in episode)
                avg_q = np.mean(self.Q)
                
                self.history.append({
                    'episode': episode_num + 1,
                    'reward': episode_reward,
                    'avg_q': avg_q,
                    'episode_length': len(episode)
                })
                
                if (episode_num + 1) % 100 == 0:
                    print(f"Épisode {episode_num + 1}: Récompense = {episode_reward:.3f}, Q moyen = {avg_q:.6f}")
        
        return {
            'Q': self.Q,
            'policy': self.target_policy,
            'history': self.history,
            'C': self.C
        }
    
    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Évalue la politique cible apprise.
        
        Args:
            num_episodes: Nombre d'épisodes d'évaluation
            
        Returns:
            Dictionnaire avec les métriques d'évaluation
        """
        if self.target_policy is None:
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
                action = self.target_policy[state]
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                steps += 1
            
            total_rewards.append(episode_reward)
            total_steps.append(steps)
            
            if episode_reward > 0:
                successes += 1
        
        return {
            'average_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'success_rate': successes / num_episodes,
            'average_steps': np.mean(total_steps),
            'std_steps': np.std(total_steps)
        }
    
    def get_action(self, state: int) -> int:
        """
        Retourne l'action selon la politique cible apprise.
        
        Args:
            state: État courant
            
        Returns:
            Action à prendre
        """
        if self.target_policy is None:
            raise ValueError("Aucune politique disponible. Entraînez d'abord l'algorithme.")
        
        return self.target_policy[state]
    
    def save(self, filepath: str) -> None:
        """
        Sauvegarde le modèle.
        
        Args:
            filepath: Chemin de sauvegarde
        """
        model_data = {
            'Q': self.Q,
            'target_policy': self.target_policy,
            'C': self.C,
            'history': self.history,
            'gamma': self.gamma,
            'epsilon': self.epsilon
        }
        save_model(model_data, filepath)
        
    def load(self, filepath: str) -> None:
        """
        Charge le modèle.
        
        Args:
            filepath: Chemin du modèle
        """
        model_data = load_model(filepath)
        self.Q = model_data['Q']
        self.target_policy = model_data['target_policy']
        self.C = model_data['C']
        self.history = model_data['history']
        self.gamma = model_data['gamma']
        self.epsilon = model_data['epsilon'] 