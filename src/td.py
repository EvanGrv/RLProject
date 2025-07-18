"""
Module de Temporal Difference pour le Reinforcement Learning.

Ce module implémente les algorithmes de différence temporelle :
- Sarsa (On-policy TD control)
- Q-Learning (Off-policy TD control)
- Expected Sarsa (Variation de Sarsa avec expectation)

Ces algorithmes apprennent à partir d'expériences individuelles
sans attendre la fin d'un épisode.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from collections import defaultdict
from src.utils_io import save_model, load_model


class Sarsa:
    """
    Implémentation de l'algorithme Sarsa.
    
    Sarsa est un algorithme on-policy qui met à jour les valeurs Q
    en utilisant l'action réellement prise par la politique.
    """
    
    def __init__(self, env: Any, alpha: float = 0.01, gamma: float = 0.9999, epsilon: float = 1.0):
        """
        Initialise Sarsa.
        
        Args:
            env: Environnement
            alpha: Taux d'apprentissage
            gamma: Facteur de discount
            epsilon: Paramètre epsilon-greedy initial
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.Q = None
        self.policy = None
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
        
        # Initialiser la table Q
        self.Q = np.zeros((self.nS, self.nA), dtype=np.float64)
        
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
        
    def epsilon_greedy_policy(self, state: int) -> int:
        """
        Politique epsilon-greedy.
        
        Args:
            state: État actuel
            
        Returns:
            Action sélectionnée
        """
        if np.random.uniform(0., 1.) <= self.epsilon:
            # Exploration : action aléatoire
            return np.random.randint(0, self.nA)
        else:
            # Exploitation : meilleure action selon Q
            return int(np.argmax(self.Q[state, :]))
        
    def update_q_value(self, state: int, action: int, reward: float, 
                      next_state: int, next_action: int):
        """
        Met à jour la valeur Q selon l'équation Sarsa.
        
        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            next_action: Action suivante
        """
        if next_state in self.terminal_states:
            # État terminal : Q(s',a') = 0
            target = reward
        else:
            # État non terminal : Q(s',a') selon l'action réellement prise
            target = reward + self.gamma * self.Q[next_state, next_action]
        
        # Mise à jour Sarsa : Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        
        return td_error
        
    def train_episode(self) -> Dict[str, Any]:
        """
        Entraîne sur un épisode complet.
        
        Returns:
            Statistiques de l'épisode
        """
        # Initialiser l'état
        state = np.random.randint(0, self.nS)
        if state in self.terminal_states:
            # Éviter de commencer dans un état terminal
            non_terminal_states = [s for s in range(self.nS) if s not in self.terminal_states]
            if non_terminal_states:
                state = np.random.choice(non_terminal_states)
        
        # Choisir l'action initiale
        action = self.epsilon_greedy_policy(state)
        
        total_reward = 0.0
        total_loss = 0.0
        steps = 0
        
        # Boucle de l'épisode
        while state not in self.terminal_states and steps < 1000:  # Protection contre boucles infinies
            # Prendre l'action et observer s', r
            next_state, reward = self.sample_next_state_reward(state, action)
            
            if next_state in self.terminal_states:
                # État terminal : mise à jour finale
                td_error = self.update_q_value(state, action, reward, next_state, 0)
                total_reward += reward
                total_loss += td_error ** 2
                steps += 1
                break
            else:
                # Choisir l'action suivante a' selon la politique epsilon-greedy
                next_action = self.epsilon_greedy_policy(next_state)
                
                # Mise à jour Sarsa
                td_error = self.update_q_value(state, action, reward, next_state, next_action)
                
                # Préparer pour l'itération suivante
                state = next_state
                action = next_action
                total_reward += reward
                total_loss += td_error ** 2
                steps += 1
        
        avg_loss = total_loss / steps if steps > 0 else 0.0
        
        return {
            'episode_reward': total_reward,
            'episode_loss': avg_loss,
            'episode_steps': steps,
            'epsilon': self.epsilon
        }
        
    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Sarsa.
        
        Args:
            num_episodes: Nombre d'épisodes d'entraînement
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        self.history = []
        
        # Epsilon decay
        epsilon_decay_step = self.initial_epsilon / num_episodes
        
        # Variables pour EMA (Exponential Moving Average)
        ema_score = 0.0
        ema_loss = 0.0
        ema_count = 0
        
        all_ema_scores = []
        all_ema_losses = []
        epsilon_values = []
        
        print(f"Démarrage de l'entraînement Sarsa pour {num_episodes} épisodes...")
        
        for episode in range(1, num_episodes + 1):
            # Entraîner sur un épisode
            episode_stats = self.train_episode()
            
            # Mettre à jour EMA
            ema_score = 0.95 * ema_score + (1 - 0.95) * episode_stats['episode_reward']
            ema_loss = 0.95 * ema_loss + (1 - 0.95) * episode_stats['episode_loss']
            ema_count += 1
            
            # Correction du biais EMA
            corrected_ema_score = ema_score / (1 - 0.95 ** ema_count)
            corrected_ema_loss = ema_loss / (1 - 0.95 ** ema_count)
            
            # Décroissance d'epsilon
            self.epsilon = max(0.01, self.epsilon - epsilon_decay_step)  # Minimum epsilon = 0.01
            
            # Enregistrer les statistiques
            self.history.append({
                'episode': episode,
                'episode_reward': episode_stats['episode_reward'],
                'episode_loss': episode_stats['episode_loss'],
                'episode_steps': episode_stats['episode_steps'],
                'ema_score': corrected_ema_score,
                'ema_loss': corrected_ema_loss,
                'epsilon': self.epsilon
            })
            
            all_ema_scores.append(corrected_ema_score)
            all_ema_losses.append(corrected_ema_loss)
            epsilon_values.append(self.epsilon)
            
            # Affichage périodique
            if episode % max(1, num_episodes // 20) == 0 or episode == 1:
                print(f"Épisode {episode}/{num_episodes} - "
                      f"Reward EMA: {corrected_ema_score:.4f}, "
                      f"Loss EMA: {corrected_ema_loss:.4f}, "
                      f"Epsilon: {self.epsilon:.4f}")
        
        # Extraire la politique finale (greedy par rapport à Q)
        self.policy = np.argmax(self.Q, axis=1)
        
        print(f"Entraînement terminé ! Epsilon final: {self.epsilon:.4f}")
        
        return {
            'Q': self.Q,
            'policy': self.policy,
            'history': self.history,
            'episodes': num_episodes,
            'all_ema_scores': all_ema_scores,
            'all_ema_losses': all_ema_losses,
            'epsilon_values': epsilon_values,
            'final_epsilon': self.epsilon
        }
        
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        model_data = {
            'Q': self.Q,
            'policy': self.policy,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'initial_epsilon': self.initial_epsilon,
            'history': self.history
        }
        save_model(model_data, filepath)
        
    def load(self, filepath: str):
        """Charge un modèle sauvegardé."""
        model_data = load_model(filepath)
        self.Q = model_data['Q']
        self.policy = model_data['policy']
        self.alpha = model_data['alpha']
        self.gamma = model_data['gamma']
        self.epsilon = model_data['epsilon']
        self.initial_epsilon = model_data.get('initial_epsilon', self.epsilon)
        self.history = model_data['history']


class QLearning:
    """
    Implémentation de l'algorithme Q-Learning.
    
    Q-Learning est un algorithme off-policy qui met à jour les valeurs Q
    en utilisant l'action optimale (max) indépendamment de la politique suivie.
    """
    
    def __init__(self, env: Any, alpha: float = 0.01, gamma: float = 0.999, epsilon: float = 1.0):
        """
        Initialise Q-Learning.
        
        Args:
            env: Environnement
            alpha: Taux d'apprentissage
            gamma: Facteur de discount
            epsilon: Paramètre epsilon-greedy initial
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.Q = None
        self.policy = None
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
        
        # Initialiser la table Q
        self.Q = np.zeros((self.nS, self.nA), dtype=np.float64)
        
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
        
    def epsilon_greedy_policy(self, state: int) -> int:
        """
        Politique epsilon-greedy.
        
        Args:
            state: État actuel
            
        Returns:
            Action sélectionnée
        """
        if np.random.rand() < self.epsilon:
            # Exploration : action aléatoire
            return np.random.randint(0, self.nA)
        else:
            # Exploitation : action qui maximise Q
            return int(np.argmax(self.Q[state, :]))
        
    def greedy_policy(self, state: int) -> int:
        """
        Politique greedy (pour l'évaluation).
        
        Args:
            state: État actuel
            
        Returns:
            Action optimale
        """
        return int(np.argmax(self.Q[state, :]))
        
    def update_q_value(self, state: int, action: int, reward: float, next_state: int):
        """
        Met à jour la valeur Q selon l'équation Q-Learning.
        
        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
        """
        if next_state in self.terminal_states:
            # État terminal : max Q(s',a') = 0
            target = reward
        else:
            # État non terminal : max_a' Q(s',a')
            q_next = float(np.max(self.Q[next_state, :]))
            target = reward + self.gamma * q_next
        
        # Mise à jour Q-Learning : Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        
        return td_error
        
    def train_episode(self) -> Dict[str, Any]:
        """
        Entraîne sur un épisode complet.
        
        Returns:
            Statistiques de l'épisode
        """
        # Initialiser l'état
        state = np.random.randint(0, self.nS)
        if state in self.terminal_states:
            # Éviter de commencer dans un état terminal
            non_terminal_states = [s for s in range(self.nS) if s not in self.terminal_states]
            if non_terminal_states:
                state = np.random.choice(non_terminal_states)
        
        total_reward = 0.0
        total_loss = 0.0
        steps = 0
        
        # Boucle de l'épisode
        while state not in self.terminal_states and steps < 1000:  # Protection contre boucles infinies
            # Choisir l'action selon epsilon-greedy
            action = self.epsilon_greedy_policy(state)
            
            # Prendre l'action et observer s', r
            next_state, reward = self.sample_next_state_reward(state, action)
            
            # Mise à jour Q-Learning (off-policy)
            td_error = self.update_q_value(state, action, reward, next_state)
            
            # Mise à jour des statistiques
            total_reward += reward
            total_loss += td_error ** 2
            steps += 1
            
            # Transition vers l'état suivant
            state = next_state
        
        avg_loss = total_loss / steps if steps > 0 else 0.0
        
        return {
            'episode_reward': total_reward,
            'episode_loss': avg_loss,
            'episode_steps': steps,
            'epsilon': self.epsilon
        }
        
    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Q-Learning.
        
        Args:
            num_episodes: Nombre d'épisodes d'entraînement
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        self.history = []
        
        # Epsilon decay
        epsilon_decay = self.initial_epsilon / num_episodes
        
        # Variables pour EMA (Exponential Moving Average)
        ema_score = 0.0
        ema_loss = 0.0
        ema_score_count = 0
        ema_loss_count = 0
        
        all_ema_scores = []
        all_ema_losses = []
        epsilon_values = []
        
        print(f"Démarrage de l'entraînement Q-Learning pour {num_episodes} épisodes...")
        
        for episode in range(1, num_episodes + 1):
            # Entraîner sur un épisode
            episode_stats = self.train_episode()
            
            # Mettre à jour EMA
            ema_loss = 0.95 * ema_loss + (1 - 0.95) * episode_stats['episode_loss']
            ema_score = 0.95 * ema_score + (1 - 0.95) * episode_stats['episode_reward']
            ema_loss_count += 1
            ema_score_count += 1
            
            # Correction du biais EMA
            corrected_ema_loss = ema_loss / (1 - 0.95 ** ema_loss_count)
            corrected_ema_score = ema_score / (1 - 0.95 ** ema_score_count)
            
            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon - epsilon_decay)  # Minimum epsilon = 0.01
            
            # Enregistrer les statistiques
            self.history.append({
                'episode': episode,
                'episode_reward': episode_stats['episode_reward'],
                'episode_loss': episode_stats['episode_loss'],
                'episode_steps': episode_stats['episode_steps'],
                'ema_score': corrected_ema_score,
                'ema_loss': corrected_ema_loss,
                'epsilon': self.epsilon
            })
            
            all_ema_scores.append(corrected_ema_score)
            all_ema_losses.append(corrected_ema_loss)
            epsilon_values.append(self.epsilon)
            
            # Affichage périodique
            if episode % max(1, num_episodes // 20) == 0 or episode == 1:
                print(f"Épisode {episode}: EMA Loss={corrected_ema_loss:.4f}, "
                      f"EMA Score={corrected_ema_score:.4f}, Epsilon={self.epsilon:.4f}")
        
        # Extraire la politique finale (greedy par rapport à Q)
        self.policy = np.argmax(self.Q, axis=1)
        
        print(f"Entraînement Q-Learning terminé ! Epsilon final: {self.epsilon:.4f}")
        
        return {
            'Q': self.Q,
            'policy': self.policy,
            'history': self.history,
            'episodes': num_episodes,
            'all_ema_scores': all_ema_scores,
            'all_ema_losses': all_ema_losses,
            'epsilon_values': epsilon_values,
            'final_epsilon': self.epsilon
        }
        
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        model_data = {
            'Q': self.Q,
            'policy': self.policy,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'initial_epsilon': self.initial_epsilon,
            'history': self.history
        }
        save_model(model_data, filepath)
        
    def load(self, filepath: str):
        """Charge un modèle sauvegardé."""
        model_data = load_model(filepath)
        self.Q = model_data['Q']
        self.policy = model_data['policy']
        self.alpha = model_data['alpha']
        self.gamma = model_data['gamma']
        self.epsilon = model_data['epsilon']
        self.initial_epsilon = model_data.get('initial_epsilon', self.epsilon)
        self.history = model_data['history']


class ExpectedSarsa:
    """
    Implémentation de l'algorithme Expected Sarsa.
    
    Expected Sarsa est une variation de Sarsa qui utilise l'espérance
    de Q(s', a') sous la politique epsilon-greedy au lieu de Q(s', a')
    pour l'action suivante réellement prise.
    """
    
    def __init__(self, env: Any, alpha: float = 0.01, gamma: float = 0.999, epsilon: float = 1.0):
        """
        Initialise Expected Sarsa.
        
        Args:
            env: Environnement
            alpha: Taux d'apprentissage
            gamma: Facteur de discount
            epsilon: Paramètre epsilon-greedy initial
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.Q = None
        self.policy = None
        self.history = []
        
        # Initialiser les structures MDP
        self._initialize_mdp_structures()
        
    def _initialize_mdp_structures(self):
        """Initialise les structures MDP nécessaires."""
        # Détection du type d'environnement
        if hasattr(self.env, 'observation_space') and hasattr(self.env, 'action_space'):
            # Environnement Gym
            self.nS = self.env.observation_space.n
            self.nA = self.env.action_space.n
        else:
            # Environnement personnalisé
            self.nS = getattr(self.env, 'nS', 100)
            self.nA = getattr(self.env, 'nA', 4)
        
        # Initialiser Q avec des zéros
        self.Q = np.zeros((self.nS, self.nA))
        
        # Détecter les états terminaux
        self.terminal_states = set()
        if hasattr(self.env, 'P'):
            for s in range(self.nS):
                for a in range(self.nA):
                    if s in self.env.P and a in self.env.P[s]:
                        for prob, next_state, reward, done in self.env.P[s][a]:
                            if done:
                                self.terminal_states.add(next_state)
        
        # Extraire les probabilités de transition et récompenses
        self.transition_probs = None
        self.rewards = None
        if hasattr(self.env, 'P'):
            self.transition_probs = self._extract_transition_probs()
            self.rewards = self._extract_rewards()
        
    def _extract_transition_probs(self):
        """Extrait les probabilités de transition de l'environnement."""
        # Trouver toutes les récompenses possibles
        all_rewards = set()
        for s in range(self.nS):
            for a in range(self.nA):
                if s in self.env.P and a in self.env.P[s]:
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        all_rewards.add(reward)
        
        rewards = sorted(list(all_rewards))
        p = np.zeros((self.nS, self.nA, self.nS, len(rewards)))
        
        for s in range(self.nS):
            for a in range(self.nA):
                if s in self.env.P and a in self.env.P[s]:
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        r_idx = rewards.index(reward)
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
        
    def epsilon_greedy_policy(self, state: int) -> int:
        """
        Politique epsilon-greedy.
        
        Args:
            state: État actuel
            
        Returns:
            Action sélectionnée
        """
        if np.random.uniform(0., 1.) <= self.epsilon:
            # Exploration : action aléatoire
            return np.random.randint(0, self.nA)
        else:
            # Exploitation : meilleure action selon Q
            return int(np.argmax(self.Q[state, :]))
    
    def compute_expected_q_value(self, state: int) -> float:
        """
        Calcule l'espérance de Q(s, a) sous la politique epsilon-greedy.
        
        Args:
            state: État pour lequel calculer l'espérance
            
        Returns:
            Espérance de Q(s, a)
        """
        if state in self.terminal_states:
            return 0.0
        
        # Obtenir les Q-values pour toutes les actions
        q_values = self.Q[state, :]
        
        # Déterminer l'action greedy (meilleure action)
        greedy_action = int(np.argmax(q_values))
        
        # Calculer les probabilités de la politique epsilon-greedy
        policy_probs = np.ones(self.nA) * (self.epsilon / self.nA)
        policy_probs[greedy_action] += (1.0 - self.epsilon)
        
        # Calculer l'espérance : E[Q(s, a)] = Σ P(a|s) * Q(s, a)
        expected_q = np.dot(policy_probs, q_values)
        
        return float(expected_q)
        
    def update_q_value(self, state: int, action: int, reward: float, next_state: int) -> float:
        """
        Met à jour la valeur Q selon l'équation Expected Sarsa.
        
        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            
        Returns:
            Erreur TD
        """
        if next_state in self.terminal_states:
            # État terminal : espérance = 0
            target = reward
        else:
            # État non terminal : utiliser l'espérance de Q(s', a') sous epsilon-greedy
            expected_q = self.compute_expected_q_value(next_state)
            target = reward + self.gamma * expected_q
        
        # Mise à jour Expected Sarsa : Q(s,a) ← Q(s,a) + α[r + γE[Q(s',a')] - Q(s,a)]
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        
        return td_error
        
    def train_episode(self) -> Dict[str, Any]:
        """
        Entraîne sur un épisode complet.
        
        Returns:
            Statistiques de l'épisode
        """
        # Initialiser l'état
        state = np.random.randint(0, self.nS)
        if state in self.terminal_states:
            # Éviter de commencer dans un état terminal
            non_terminal_states = [s for s in range(self.nS) if s not in self.terminal_states]
            if non_terminal_states:
                state = np.random.choice(non_terminal_states)
        
        total_reward = 0.0
        total_loss = 0.0
        steps = 0
        
        # Boucle de l'épisode
        while state not in self.terminal_states and steps < 1000:  # Protection contre boucles infinies
            # Choisir l'action selon epsilon-greedy
            action = self.epsilon_greedy_policy(state)
            
            # Prendre l'action et observer s', r
            next_state, reward = self.sample_next_state_reward(state, action)
            
            # Mise à jour Expected Sarsa
            td_error = self.update_q_value(state, action, reward, next_state)
            
            # Accumuler les statistiques
            total_reward += reward
            total_loss += td_error ** 2
            steps += 1
            
            # Transition vers l'état suivant
            state = next_state
        
        avg_loss = total_loss / steps if steps > 0 else 0.0
        
        return {
            'episode_reward': total_reward,
            'episode_loss': avg_loss,
            'episode_steps': steps,
            'epsilon': self.epsilon
        }
        
    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Expected Sarsa.
        
        Args:
            num_episodes: Nombre d'épisodes d'entraînement
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        self.history = []
        
        # Epsilon decay
        epsilon_decay_step = self.initial_epsilon / num_episodes
        
        # Variables pour EMA (Exponential Moving Average)
        ema_score = 0.0
        ema_loss = 0.0
        ema_count = 0
        
        all_ema_scores = []
        all_ema_losses = []
        epsilon_values = []
        
        print(f"Démarrage de l'entraînement Expected Sarsa pour {num_episodes} épisodes...")
        
        for episode in range(1, num_episodes + 1):
            # Entraîner sur un épisode
            episode_stats = self.train_episode()
            
            # Mettre à jour EMA
            ema_score = 0.95 * ema_score + (1 - 0.95) * episode_stats['episode_reward']
            ema_loss = 0.95 * ema_loss + (1 - 0.95) * episode_stats['episode_loss']
            ema_count += 1
            
            # Correction du biais EMA
            corrected_ema_score = ema_score / (1 - 0.95 ** ema_count)
            corrected_ema_loss = ema_loss / (1 - 0.95 ** ema_count)
            
            # Décroissance d'epsilon
            self.epsilon = max(0.01, self.epsilon - epsilon_decay_step)  # Minimum epsilon = 0.01
            
            # Enregistrer les statistiques
            self.history.append({
                'episode': episode,
                'episode_reward': episode_stats['episode_reward'],
                'episode_loss': episode_stats['episode_loss'],
                'episode_steps': episode_stats['episode_steps'],
                'ema_score': corrected_ema_score,
                'ema_loss': corrected_ema_loss,
                'epsilon': self.epsilon
            })
            
            # Stocker pour les graphiques
            all_ema_scores.append(corrected_ema_score)
            all_ema_losses.append(corrected_ema_loss)
            epsilon_values.append(self.epsilon)
            
            # Affichage périodique
            if episode % max(1, num_episodes // 10) == 0:
                print(f"Épisode {episode}: EMA Loss={corrected_ema_loss:.4f}, "
                      f"EMA Score={corrected_ema_score:.4f}, Epsilon={self.epsilon:.4f}")
        
        # Extraire la politique finale
        self.policy = self.extract_policy()
        
        print(f"Entraînement terminé après {num_episodes} épisodes")
        print(f"Score EMA final: {corrected_ema_score:.4f}")
        print(f"Loss EMA finale: {corrected_ema_loss:.4f}")
        
        return {
            'all_ema_scores': all_ema_scores,
            'all_ema_losses': all_ema_losses,
            'epsilon_values': epsilon_values,
            'final_policy': self.policy,
            'final_q_values': self.Q.copy(),
            'history': self.history
        }
        
    def extract_policy(self) -> np.ndarray:
        """
        Extrait la politique greedy optimale à partir des valeurs Q.
        
        Returns:
            Politique déterministe optimale
        """
        policy = np.zeros(self.nS, dtype=int)
        for s in range(self.nS):
            if s not in self.terminal_states:
                policy[s] = int(np.argmax(self.Q[s, :]))
        return policy
        
    def get_state_value(self, state: int) -> float:
        """
        Calcule la valeur d'un état selon la politique actuelle.
        
        Args:
            state: État à évaluer
            
        Returns:
            Valeur de l'état
        """
        if state in self.terminal_states:
            return 0.0
        return self.compute_expected_q_value(state)
        
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        model_data = {
            'Q': dict(enumerate(self.Q.tolist())),
            'policy': self.policy.tolist() if self.policy is not None else None,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'history': self.history
        }
        save_model(model_data, filepath)
        
    def load(self, filepath: str):
        """Charge un modèle sauvegardé."""
        model_data = load_model(filepath)
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n), model_data['Q'])
        self.policy = model_data['policy']
        self.alpha = model_data['alpha']
        self.gamma = model_data['gamma']
        self.epsilon = model_data['epsilon']
        self.history = model_data['history'] 