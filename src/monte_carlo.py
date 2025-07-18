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
        else:
            # Environnement générique
            self.nS = getattr(self.env, 'observation_space', type('obj', (object,), {'n': 16})).n
            self.nA = getattr(self.env, 'action_space', type('obj', (object,), {'n': 4})).n
        
        # Initialiser Q et politique
        self.Q = np.zeros((self.nS, self.nA), dtype=float)
        self.policy = np.zeros(self.nS, dtype=int)
        
        # Identifier les états terminaux
        self.terminal_states = []
        if hasattr(self.env, 'desc'):
            # Pour FrozenLake par exemple
            desc = self.env.desc
            for i in range(desc.shape[0]):
                for j in range(desc.shape[1]):
                    if desc[i, j] in [b'H', b'G']:  # Hole ou Goal
                        self.terminal_states.append(i * desc.shape[1] + j)
        
        # Initialiser les probabilités de transition si disponibles
        if hasattr(self.env, 'P'):
            self.transition_probs = self._extract_transition_probabilities()
            self.rewards = self._extract_rewards()
        else:
            self.transition_probs = None
            self.rewards = [0.0, 1.0]  # Récompenses par défaut
            
    def _extract_transition_probabilities(self):
        """Extrait les probabilités de transition de l'environnement."""
        rewards = set()
        for s in range(self.nS):
            for a in range(self.nA):
                if s in self.env.P and a in self.env.P[s]:
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        rewards.add(reward)
        
        self.rewards = sorted(list(rewards))
        p = np.zeros((self.nS, self.nA, self.nS, len(self.rewards)))
        
        for s in range(self.nS):
            for a in range(self.nA):
                if s in self.env.P and a in self.env.P[s]:
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        r_idx = self.rewards.index(reward)
                        p[s, a, next_state, r_idx] = prob
        
        return p
        
    def _extract_rewards(self):
        """Extrait la liste des récompenses possibles."""
        rewards = set()
        for s in range(self.nS):
            for a in range(self.nA):
                if s in self.env.P and a in self.env.P[s]:
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        rewards.add(reward)
        return sorted(list(rewards))
    
    def sample_next_state_reward(self, s: int, a: int) -> Tuple[int, float]:
        """
        Échantillonne l'état suivant s' et la récompense r à partir des probabilités de transition.
        
        Args:
            s: État actuel
            a: Action
            
        Returns:
            Tuple (next_state, reward)
        """
        if self.transition_probs is None:
            # Fallback pour environnements sans probabilités explicites
            state = self.env.reset()
            self.env.unwrapped.s = s
            next_state, reward, done, _ = self.env.step(a)
            return next_state, reward
        
        # Utiliser les probabilités de transition
        num_states = self.transition_probs.shape[2]
        num_rewards = self.transition_probs.shape[3]
        
        # Construire une liste de (s', r_idx) avec prob > 0
        candidates = []
        probs = []
        for s_p in range(num_states):
            for r_idx in range(num_rewards):
                prob = self.transition_probs[s, a, s_p, r_idx]
                if prob > 0:
                    candidates.append((s_p, r_idx))
                    probs.append(prob)
        
        # Normaliser (au cas où)
        probs = np.array(probs, dtype=np.float64)
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            # Cas d'erreur, distribution uniforme
            probs = np.ones(len(candidates)) / len(candidates) if candidates else np.array([1.0])
            
        # Tirage
        if candidates:
            idx = np.random.choice(len(candidates), p=probs)
            s_p, r_idx = candidates[idx]
            return s_p, self.rewards[r_idx]
        else:
            return s, 0.0
        
    def generate_episode(self, exploring_starts: bool = True) -> List[Tuple[int, int, float]]:
        """
        Génère un épisode complet avec Exploring Starts.
        
        Args:
            exploring_starts: Si True, utilise exploring starts
            
        Returns:
            Liste des transitions (état, action, reward)
        """
        episode = []
        
        if exploring_starts:
            # Exploring Start: état et action initiaux aléatoires
            start_state = random.choice(list(range(self.nS)))
            start_action = random.choice(list(range(self.nA)))
            
            # Appliquer la transition (s,a) fixée par ES
            s_next, r = self.sample_next_state_reward(start_state, start_action)
            episode.append((start_state, start_action, r))
            s = s_next
        else:
            # Démarrage normal
            s = random.choice(list(range(self.nS)))
        
        # Continuer selon la politique jusqu'à l'état terminal
        max_steps = 1000  # Protection contre les boucles infinies
        steps = 0
        
        while s not in self.terminal_states and steps < max_steps:
            a = self.policy[s]
            s_next, r = self.sample_next_state_reward(s, a)
            episode.append((s, a, r))
            s = s_next
            steps += 1
            
            if s in self.terminal_states:
                break
        
        return episode
        
    def update_q_values(self, episode: List[Tuple[int, int, float]]):
        """
        Met à jour les valeurs Q à partir d'un épisode (First-Visit).
        
        Args:
            episode: Liste des transitions (état, action, reward)
        """
        # Calculer les returns
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_tp1 = episode[t]
            G = self.gamma * G + r_tp1
            
            # Vérifier si c'est la première visite de (s_t, a_t)
            first_visit = True
            for j in range(t):
                if episode[j][0] == s_t and episode[j][1] == a_t:
                    first_visit = False
                    break
            
            if first_visit:
                # Mettre à jour Q(s,a) avec la moyenne des returns
                key = (s_t, a_t)
                self.returns_sum[key] = self.returns_sum.get(key, 0.0) + G
                self.returns_count[key] = self.returns_count.get(key, 0) + 1
                self.Q[s_t, a_t] = self.returns_sum[key] / self.returns_count[key]
        
    def improve_policy(self):
        """Met à jour la politique basée sur les valeurs Q (greedy)."""
        for s in range(self.nS):
            if s not in self.terminal_states:
                self.policy[s] = int(np.argmax(self.Q[s, :]))
        
    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Entraîne l'algorithme MC-ES.
        
        Args:
            num_episodes: Nombre d'épisodes d'entraînement
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        self.history = []
        
        for ep in range(1, num_episodes + 1):
            # Générer un épisode
            episode = self.generate_episode(exploring_starts=True)
            
            # Mettre à jour les valeurs Q
            self.update_q_values(episode)
            
            # Améliorer la politique
            self.improve_policy()
            
            # Enregistrer les statistiques
            if ep % max(1, num_episodes // 20) == 0 or ep == 1:
                avg_q = np.mean(self.Q)
                max_q = np.max(self.Q)
                episode_length = len(episode)
                
                self.history.append({
                    'episode': ep,
                    'avg_q_value': avg_q,
                    'max_q_value': max_q,
                    'episode_length': episode_length,
                    'num_state_actions_visited': len(self.returns_count)
                })
                
                print(f"Épisode {ep}/{num_episodes} - Q moyen: {avg_q:.4f}, Longueur: {episode_length}")
        
        return {
            'Q': self.Q,
            'policy': self.policy,
            'history': self.history,
            'episodes': num_episodes,
            'returns_sum': self.returns_sum,
            'returns_count': self.returns_count
        }
        
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        model_data = {
            'Q': self.Q,
            'policy': self.policy,
            'gamma': self.gamma,
            'history': self.history,
            'returns_sum': self.returns_sum,
            'returns_count': self.returns_count
        }
        save_model(model_data, filepath)
        
    def load(self, filepath: str):
        """Charge un modèle sauvegardé."""
        model_data = load_model(filepath)
        self.Q = model_data['Q']
        self.policy = model_data['policy']
        self.gamma = model_data['gamma']
        self.history = model_data['history']
        self.returns_sum = model_data['returns_sum']
        self.returns_count = model_data['returns_count']


class OnPolicyFirstVisitMC:
    """
    Implémentation de On-policy First-Visit Monte Carlo.
    
    Cet algorithme utilise une politique epsilon-greedy et met à jour
    les valeurs Q seulement pour la première visite de chaque état-action.
    """
    
    def __init__(self, env: Any, gamma: float = 0.999999, epsilon: float = 0.1):
        """
        Initialise On-policy First-Visit MC.
        
        Args:
            env: Environnement
            gamma: Facteur de discount
            epsilon: Paramètre epsilon-greedy
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = None
        self.policy = None  # Politique epsilon-greedy sous forme de matrice de probabilités
        self.returns_counts = None
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
        else:
            # Environnement générique
            self.nS = getattr(self.env, 'observation_space', type('obj', (object,), {'n': 16})).n
            self.nA = getattr(self.env, 'action_space', type('obj', (object,), {'n': 4})).n
        
        # Initialiser la politique epsilon-greedy uniforme
        self.policy = (1.0 / self.nA) * np.ones((self.nS, self.nA))
        
        # Initialiser Q avec des valeurs aléatoires
        self.Q = np.random.random((self.nS, self.nA))
        
        # Compteurs de retours pour calcul des moyennes
        self.returns_counts = np.zeros((self.nS, self.nA))
        
        # Identifier les états terminaux
        self.terminal_states = []
        if hasattr(self.env, 'desc'):
            # Pour FrozenLake par exemple
            desc = self.env.desc
            for i in range(desc.shape[0]):
                for j in range(desc.shape[1]):
                    if desc[i, j] in [b'H', b'G']:  # Hole ou Goal
                        self.terminal_states.append(i * desc.shape[1] + j)
        
        # Initialiser les probabilités de transition si disponibles
        if hasattr(self.env, 'P'):
            self.transition_probs = self._extract_transition_probabilities()
            self.rewards = self._extract_rewards()
        else:
            self.transition_probs = None
            self.rewards = [0.0, 1.0]
    
    def _extract_transition_probabilities(self):
        """Extrait les probabilités de transition de l'environnement."""
        rewards = set()
        for s in range(self.nS):
            for a in range(self.nA):
                if s in self.env.P and a in self.env.P[s]:
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        rewards.add(reward)
        
        self.rewards = sorted(list(rewards))
        p = np.zeros((self.nS, self.nA, self.nS, len(self.rewards)))
        
        for s in range(self.nS):
            for a in range(self.nA):
                if s in self.env.P and a in self.env.P[s]:
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        r_idx = self.rewards.index(reward)
                        p[s, a, next_state, r_idx] = prob
        
        return p
        
    def _extract_rewards(self):
        """Extrait la liste des récompenses possibles."""
        rewards = set()
        for s in range(self.nS):
            for a in range(self.nA):
                if s in self.env.P and a in self.env.P[s]:
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        rewards.add(reward)
        return sorted(list(rewards))
    
    def sample_next_state_reward(self, s: int, a: int) -> Tuple[int, float]:
        """
        Échantillonne l'état suivant et la récompense.
        
        Args:
            s: État actuel
            a: Action
            
        Returns:
            Tuple (next_state, reward)
        """
        if self.transition_probs is None:
            # Fallback pour environnements sans probabilités explicites
            state = self.env.reset()
            self.env.unwrapped.s = s
            next_state, reward, done, _ = self.env.step(a)
            return next_state, reward
        
        # Utiliser les probabilités de transition
        candidates = []
        probs = []
        for s_p in range(self.nS):
            for r_idx in range(len(self.rewards)):
                prob = self.transition_probs[s, a, s_p, r_idx]
                if prob > 0:
                    candidates.append((s_p, r_idx))
                    probs.append(prob)
        
        if candidates:
            probs = np.array(probs)
            probs /= probs.sum()
            idx = np.random.choice(len(candidates), p=probs)
            s_p, r_idx = candidates[idx]
            return s_p, self.rewards[r_idx]
        else:
            return s, 0.0
        
    def epsilon_greedy_policy(self, state: int) -> int:
        """
        Sélectionne une action selon la politique epsilon-greedy.
        
        Args:
            state: État actuel
            
        Returns:
            Action sélectionnée
        """
        all_actions = np.arange(self.nA)
        return np.random.choice(all_actions, p=self.policy[state])
        
    def generate_episode(self) -> List[Tuple[int, int, float]]:
        """
        Génère un épisode suivant la politique epsilon-greedy.
        
        Returns:
            Liste des transitions (état, action, reward)
        """
        episode = []
        
        # Démarrer depuis un état aléatoire
        s = np.random.randint(self.nS)
        
        max_steps = 1000  # Protection contre les boucles infinies
        steps = 0
        
        while s not in self.terminal_states and steps < max_steps:
            # Sélectionner une action selon la politique epsilon-greedy
            a = self.epsilon_greedy_policy(s)
            
            # Calculer la récompense et l'état suivant
            s_next, r = self.sample_next_state_reward(s, a)
            
            # Ajouter à l'épisode
            episode.append((s, a, r))
            
            # Passer à l'état suivant
            s = s_next
            steps += 1
            
            if s in self.terminal_states:
                break
        
        return episode
        
    def update_q_values(self, episode: List[Tuple[int, int, float]]):
        """
        Met à jour les valeurs Q (first-visit seulement).
        
        Args:
            episode: Liste des transitions (état, action, reward)
        """
        # Mettre les valeurs Q des états terminaux à 0
        for terminal_state in self.terminal_states:
            self.Q[terminal_state, :] = 0.0
        
        G = 0.0
        
        # Parcourir l'épisode en sens inverse
        for t in reversed(range(len(episode))):
            s_t, a_t, r_t_plus_1 = episode[t]
            
            # Calculer le return
            G = self.gamma * G + r_t_plus_1
            
            # Vérifier si c'est la première visite de (s_t, a_t)
            first_visit = True
            for j in range(t):
                if episode[j][0] == s_t and episode[j][1] == a_t:
                    first_visit = False
                    break
            
            if first_visit:
                # Mettre à jour Q avec la moyenne incrémentale
                self.Q[s_t, a_t] = (self.Q[s_t, a_t] * self.returns_counts[s_t, a_t] + G) / (self.returns_counts[s_t, a_t] + 1)
                self.returns_counts[s_t, a_t] += 1
                
                # Améliorer la politique (epsilon-greedy)
                best_a = np.argmax(self.Q[s_t, :])
                
                # Initialiser toutes les actions avec epsilon / num_actions
                self.policy[s_t, :] = self.epsilon / self.nA
                
                # Donner la probabilité maximale à la meilleure action
                self.policy[s_t, best_a] = 1.0 - self.epsilon + self.epsilon / self.nA
        
    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Entraîne l'algorithme On-policy First-Visit MC.
        
        Args:
            num_episodes: Nombre d'épisodes d'entraînement
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        self.history = []
        
        for episode_num in range(1, num_episodes + 1):
            # Générer un épisode
            episode = self.generate_episode()
            
            # Mettre à jour les valeurs Q et la politique
            self.update_q_values(episode)
            
            # Enregistrer les statistiques
            if episode_num % max(1, num_episodes // 20) == 0 or episode_num == 1:
                avg_q = np.mean(self.Q)
                max_q = np.max(self.Q)
                episode_length = len(episode)
                
                self.history.append({
                    'episode': episode_num,
                    'avg_q_value': avg_q,
                    'max_q_value': max_q,
                    'episode_length': episode_length,
                    'epsilon': self.epsilon
                })
                
                print(f"Épisode {episode_num}/{num_episodes} - Q moyen: {avg_q:.4f}, Longueur: {episode_length}")
        
        return {
            'Q': self.Q,
            'policy': self.policy,
            'history': self.history,
            'episodes': num_episodes,
            'returns_counts': self.returns_counts
        }
        
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        model_data = {
            'Q': self.Q,
            'policy': self.policy,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'history': self.history,
            'returns_counts': self.returns_counts
        }
        save_model(model_data, filepath)
        
    def load(self, filepath: str):
        """Charge un modèle sauvegardé."""
        model_data = load_model(filepath)
        self.Q = model_data['Q']
        self.policy = model_data['policy']
        self.gamma = model_data['gamma']
        self.epsilon = model_data['epsilon']
        self.history = model_data['history']
        self.returns_counts = model_data['returns_counts']


class OffPolicyMC:
    """
    Implémentation de Off-policy Monte Carlo avec Weighted Importance Sampling.
    
    Utilise une politique de comportement uniforme pour générer des épisodes
    et apprend une politique cible greedy via weighted importance sampling.
    """
    
    def __init__(self, env: Any, gamma: float = 1.0):
        """
        Initialise Off-policy MC.
        
        Args:
            env: Environnement
            gamma: Facteur de discount
        """
        self.env = env
        self.gamma = gamma
        self.Q = None
        self.C = None  # Cumulative weights pour weighted importance sampling
        self.target_policy = None  # Politique cible (greedy)
        self.behavior_policy = None  # Politique de comportement (uniforme)
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
        else:
            # Environnement générique
            self.nS = getattr(self.env, 'observation_space', type('obj', (object,), {'n': 16})).n
            self.nA = getattr(self.env, 'action_space', type('obj', (object,), {'n': 4})).n
        
        # Initialiser Q, C et politiques
        self.Q = np.zeros((self.nS, self.nA), dtype=np.float64)
        self.C = np.zeros((self.nS, self.nA), dtype=np.float64)  # Poids cumulatifs
        self.target_policy = np.zeros(self.nS, dtype=int)  # Politique cible (greedy)
        self.behavior_policy = np.ones((self.nS, self.nA)) / self.nA  # Politique uniforme
        
        # Identifier les états terminaux
        self.terminal_states = []
        if hasattr(self.env, 'desc'):
            # Pour FrozenLake par exemple
            desc = self.env.desc
            for i in range(desc.shape[0]):
                for j in range(desc.shape[1]):
                    if desc[i, j] in [b'H', b'G']:  # Hole ou Goal
                        self.terminal_states.append(i * desc.shape[1] + j)
        
        # Initialiser les probabilités de transition si disponibles
        if hasattr(self.env, 'P'):
            self.transition_probs = self._extract_transition_probabilities()
            self.rewards = self._extract_rewards()
        else:
            self.transition_probs = None
            self.rewards = [0.0, 1.0]
    
    def _extract_transition_probabilities(self):
        """Extrait les probabilités de transition de l'environnement."""
        rewards = set()
        for s in range(self.nS):
            for a in range(self.nA):
                if s in self.env.P and a in self.env.P[s]:
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        rewards.add(reward)
        
        self.rewards = sorted(list(rewards))
        p = np.zeros((self.nS, self.nA, self.nS, len(self.rewards)))
        
        for s in range(self.nS):
            for a in range(self.nA):
                if s in self.env.P and a in self.env.P[s]:
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        r_idx = self.rewards.index(reward)
                        p[s, a, next_state, r_idx] = prob
        
        return p
        
    def _extract_rewards(self):
        """Extrait la liste des récompenses possibles."""
        rewards = set()
        for s in range(self.nS):
            for a in range(self.nA):
                if s in self.env.P and a in self.env.P[s]:
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        rewards.add(reward)
        return sorted(list(rewards))
    
    def sample_next_state_reward(self, s: int, a: int) -> Tuple[int, float]:
        """
        Échantillonne l'état suivant et la récompense.
        
        Args:
            s: État actuel
            a: Action
            
        Returns:
            Tuple (next_state, reward)
        """
        if self.transition_probs is None:
            # Fallback pour environnements sans probabilités explicites
            state = self.env.reset()
            self.env.unwrapped.s = s
            next_state, reward, done, _ = self.env.step(a)
            return next_state, reward
        
        # Utiliser les probabilités de transition
        candidates = []
        probs = []
        for s_p in range(self.nS):
            for r_idx in range(len(self.rewards)):
                prob = self.transition_probs[s, a, s_p, r_idx]
                if prob > 0:
                    candidates.append((s_p, r_idx))
                    probs.append(prob)
        
        if candidates:
            probs = np.array(probs, dtype=np.float64)
            probs /= probs.sum()
            idx = np.random.choice(len(candidates), p=probs)
            s_p, r_idx = candidates[idx]
            return s_p, self.rewards[r_idx]
        else:
            return s, 0.0
        
    def behavior_policy_action(self, state: int) -> int:
        """
        Politique de comportement (uniforme).
        
        Args:
            state: État actuel
            
        Returns:
            Action sélectionnée
        """
        return np.random.choice(self.nA)
        
    def target_policy_action(self, state: int) -> int:
        """
        Politique cible (greedy par rapport à Q).
        
        Args:
            state: État actuel
            
        Returns:
            Action sélectionnée
        """
        return self.target_policy[state]
        
    def generate_episode(self) -> List[Tuple[int, int, float]]:
        """
        Génère un épisode avec la politique de comportement uniforme.
        
        Returns:
            Liste des transitions (état, action, reward)
        """
        episode = []
        
        # Sélection de l'état initial (non terminal)
        non_terminal_states = [s for s in range(self.nS) if s not in self.terminal_states]
        if non_terminal_states:
            s = np.random.choice(non_terminal_states)
        else:
            s = np.random.randint(self.nS)
        
        max_steps = 1000  # Protection contre les boucles infinies
        steps = 0
        
        # Tant que non terminal
        while s not in self.terminal_states and steps < max_steps:
            # Choix aléatoire pour la politique de comportement (uniforme)
            a = self.behavior_policy_action(s)
            
            # Transition
            s_next, r = self.sample_next_state_reward(s, a)
            episode.append((s, a, r))
            
            s = s_next
            steps += 1
            
            if s in self.terminal_states:
                break
        
        return episode
        
    def update_q_values(self, episode: List[Tuple[int, int, float]]):
        """
        Met à jour les valeurs Q avec weighted importance sampling.
        
        Args:
            episode: Liste des transitions (état, action, reward)
        """
        if not episode:
            return
            
        # Calcul des retours G_t et pondération W (weighted importance sampling)
        G = 0.0
        W = 1.0
        
        # Parcourir l'épisode à l'envers
        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_t_plus_1 = episode[t]
            
            # Calculer le return
            G = self.gamma * G + r_t_plus_1
            
            # Mettre à jour C et Q pour (s_t, a_t) avec weighted importance sampling
            self.C[s_t, a_t] += W
            if self.C[s_t, a_t] > 0:
                self.Q[s_t, a_t] += (W / self.C[s_t, a_t]) * (G - self.Q[s_t, a_t])
            
            # Mettre à jour la politique cible : greedy w.r.t Q[s_t]
            self.target_policy[s_t] = int(np.argmax(self.Q[s_t, :]))
            
            # Calculer le ratio d'importance sampling pour la prochaine étape
            # Politique de comportement : uniforme (probabilité 1/nA)
            # Politique cible : déterministe (probabilité 1 si action = target_policy[s_t], 0 sinon)
            behavior_prob = 1.0 / self.nA
            target_prob = 1.0 if a_t == self.target_policy[s_t] else 0.0
            
            # Si la politique cible ne sélectionnerait pas cette action, W devient 0
            if target_prob == 0.0:
                break  # W deviendrait 0, on arrête la boucle
            
            # Mise à jour du poids W
            W *= target_prob / behavior_prob
        
    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Off-policy MC.
        
        Args:
            num_episodes: Nombre d'épisodes d'entraînement
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        self.history = []
        
        for episode_num in range(1, num_episodes + 1):
            # Générer un épisode avec la politique de comportement
            episode = self.generate_episode()
            
            # Mettre à jour les valeurs Q et la politique cible
            self.update_q_values(episode)
            
            # Enregistrer les statistiques
            if episode_num % max(1, num_episodes // 20) == 0 or episode_num == 1:
                avg_q = np.mean(self.Q)
                max_q = np.max(self.Q)
                episode_length = len(episode)
                
                self.history.append({
                    'episode': episode_num,
                    'avg_q_value': avg_q,
                    'max_q_value': max_q,
                    'episode_length': episode_length,
                    'cumulative_weights_sum': np.sum(self.C)
                })
                
                print(f"Épisode {episode_num}/{num_episodes} - Q moyen: {avg_q:.4f}, Longueur: {episode_length}")
        
        return {
            'Q': self.Q,
            'target_policy': self.target_policy,
            'behavior_policy': self.behavior_policy,
            'C': self.C,
            'history': self.history,
            'episodes': num_episodes
        }
        
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        model_data = {
            'Q': self.Q,
            'C': self.C,
            'target_policy': self.target_policy,
            'behavior_policy': self.behavior_policy,
            'gamma': self.gamma,
            'history': self.history
        }
        save_model(model_data, filepath)
        
    def load(self, filepath: str):
        """Charge un modèle sauvegardé."""
        model_data = load_model(filepath)
        self.Q = model_data['Q']
        self.C = model_data['C']
        self.target_policy = model_data['target_policy']
        self.behavior_policy = model_data['behavior_policy']
        self.gamma = model_data['gamma']
        self.history = model_data['history'] 