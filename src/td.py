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


import numpy as np
import random
from collections import defaultdict
from typing import Any, List, Tuple, Dict

# (Supposons que save_model et load_model sont définis dans vos utilitaires)
# from utils import save_model, load_model

class Sarsa:
    """
    SARSA - On-policy TD Control.

    Met à jour Q(s,a) en utilisant l'action réellement prise par la politique ε-greedy.
    """

    def __init__(
        self,
        env: Any,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
    ):
        """
        Args:
            env: Environnement Gym-like (avec env.reset(), env.step())
            alpha: taux d'apprentissage ∈ (0,1]
            gamma: facteur d'actualisation ∈ [0,1]
            epsilon: paramètre initial ε pour la politique ε-greedy
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon

        # Structures MDP
        self._init_mdp_structures()

        # Pour suivre récompenses, pertes, ε
        self.history: List[Dict[str, float]] = []

    def _init_mdp_structures(self):
        """Détecte nS, nA et initialise Q et policy."""
        if hasattr(self.env, 'nS') and hasattr(self.env, 'nA'):
            self.nS, self.nA = self.env.nS, self.env.nA
        else:
            # Gym standard
            self.nS = getattr(self.env.observation_space, 'n', None)
            self.nA = getattr(self.env.action_space,    'n', None)
            if self.nS is None or self.nA is None:
                raise ValueError("Impossible de déterminer nS ou nA.")
        # Q(s,a) initialisé à zéro
        self.Q = np.zeros((self.nS, self.nA), dtype=float)
        # Politique gloutonne par défaut (sera extraite après training)
        self.policy = np.zeros(self.nS, dtype=int)

    def epsilon_greedy(self, state: int) -> int:
        """
        Sélectionne une action selon une stratégie ε-greedy sur Q[state]:
        - avec prob. ε: action aléatoire
        - sinon: argmax_a Q(state,a) (tie-breaking aléatoire)
        """
        if random.random() < self.epsilon:
            return random.randrange(self.nA)
        q_s = self.Q[state]
        max_q = np.max(q_s)
        best_actions = np.flatnonzero(q_s == max_q)
        return int(random.choice(best_actions))

    def train_episode(self) -> Dict[str, float]:
        """
        Exécute un épisode complet selon SARSA:
        1) Initialise s ← env.reset()
        2) Choisit a ← ε-greedy(s)
        3) Pour chaque pas t:
             observe r, s' ← env.step(a)
             choisit a' ← ε-greedy(s')
             met à jour Q(s,a):
               Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') − Q(s,a) ]
             (s,a) ← (s',a')
        4) Termine quand s' est terminal
        """
        # 1) état initial
        obs = self.env.reset()
        state = obs if not hasattr(self.env, 'state') else self.env.state

        # 2) action initiale
        action = self.epsilon_greedy(state)

        total_reward = 0.0
        total_td_error2 = 0.0
        steps = 0

        # boucle jusqu'à état terminal ou plafond d'étapes
        done = False
        while not done and steps < 1000:
            # prendre l'action et observer
            next_obs, reward, done, _ = self.env.step(action)
            next_state = next_obs if not hasattr(self.env, 'state') else self.env.state
            total_reward += reward

            # choisir action suivante selon policy ε-greedy
            next_action = self.epsilon_greedy(next_state)

            # 3) mise à jour SARSA
            # Formule TD-target:    r + γ Q(s',a')
            # TD-error δ = target − Q(s,a)
            td_target = reward + self.gamma * self.Q[next_state, next_action]
            td_error  = td_target - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error

            total_td_error2 += td_error**2

            # passer au prochain pas
            state, action = next_state, next_action
            steps += 1

        # moyenne quadratique des TD-errors
        avg_loss = total_td_error2 / steps if steps>0 else 0.0

        return {
            'reward': total_reward,
            'loss':   avg_loss,
            'steps':  steps
        }

    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Entraînement complet SARSA sur num_episodes épisodes.
        Décroît ε linéairement jusqu'à ε_min=0.01.
        """
        self.history.clear()
        epsilon_decay = (self.initial_epsilon - 0.01) / num_episodes

        for ep in range(1, num_episodes+1):
            stats = self.train_episode()

            # décay de ε
            self.epsilon = max(0.01, self.epsilon - epsilon_decay)

            # suivre les métriques
            self.history.append({
                'episode':   ep,
                'reward':    stats['reward'],
                'loss':      stats['loss'],
                'steps':     stats['steps'],
                'epsilon':   self.epsilon
            })
            # log périodique
            if ep % max(1, num_episodes//10) == 0:
                print(f"[SARSA] Ep {ep}/{num_episodes} │ R={stats['reward']:.2f} │ loss={stats['loss']:.4f} │ ε={self.epsilon:.3f}")

        # extraire politique gloutonne finale
        for s in range(self.nS):
            self.policy[s] = int(np.argmax(self.Q[s]))

        return {
            'Q':       self.Q,
            'policy':  self.policy,
            'history': self.history
        }

    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Évalue la politique gloutonne (ε=0) pendant num_episodes épisodes.
        """
        old_eps = self.epsilon
        self.epsilon = 0.0

        rewards, lengths, success = [], [], 0
        for _ in range(num_episodes):
            obs = self.env.reset()
            state = obs if not hasattr(self.env, 'state') else self.env.state
            done, G, steps = False, 0.0, 0

            while not done and steps<1000:
                a = self.policy[state]
                next_obs, r, done, _ = self.env.step(a)
                state = next_obs if not hasattr(self.env, 'state') else self.env.state
                G += r; steps += 1
            rewards.append(G); lengths.append(steps)
            if G>0: success += 1

        self.epsilon = old_eps
        return {
            'avg_reward':    np.mean(rewards),
            'std_reward':    np.std(rewards),
            'success_rate':  success/num_episodes,
            'avg_steps':     np.mean(lengths)
        }

    def save(self, filepath: str):
        save_model({
            'Q':      self.Q,
            'policy': self.policy,
            'alpha':  self.alpha,
            'gamma':  self.gamma,
            'epsilon': self.epsilon,
            'history': self.history
        }, filepath)

    def load(self, filepath: str):
        data = load_model(filepath)
        self.Q       = data['Q']
        self.policy  = data['policy']
        self.alpha   = data['alpha']
        self.gamma   = data['gamma']
        self.epsilon = data['epsilon']
        self.history = data['history']


class QLearning:
    """
    Q-Learning (Off-policy TD control).

    Utilise la règle de mise à jour :
      Q(s,a) ← Q(s,a) + α [ r + γ max_{a'} Q(s',a') − Q(s,a) ]
    avec une politique ε-greedy pour explorer.
    """

    def __init__(
        self,
        env: Any,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
    ):
        """
        Args:
            env: Environnement Gym-like
            alpha: pas d'apprentissage ∈ (0,1]
            gamma: facteur d'actualisation ∈ [0,1]
            epsilon: ε initial pour ε-greedy
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon

        # Détection du nombre d'états et d'actions
        if hasattr(env, 'nS') and hasattr(env, 'nA'):
            self.nS, self.nA = env.nS, env.nA
        else:
            self.nS = env.observation_space.n
            self.nA = env.action_space.n

        # Table Q(s,a) initialisée à zéro
        self.Q = np.zeros((self.nS, self.nA), dtype=float)
        # Politique gloutonne construite après entraînement
        self.policy = np.zeros(self.nS, dtype=int)

        # Historique pour suivi des métriques
        self.history: List[Dict[str, float]] = []

    def epsilon_greedy(self, state: int) -> int:
        """Sélectionne une action selon ε-greedy sur Q[state]."""
        if random.random() < self.epsilon:
            return random.randrange(self.nA)
        q_s = self.Q[state]
        max_q = np.max(q_s)
        best_actions = np.flatnonzero(q_s == max_q)
        return int(random.choice(best_actions))

    def train_episode(self) -> Dict[str, float]:
        """
        Exécute un épisode complet selon Q-Learning:
        1) s ← env.reset()
        2) Tant que non terminal:
             a ← ε-greedy(s)
             r, s' ← env.step(a)
             Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
             s ← s'
        """
        state = self.env.reset()
        total_reward = 0.0
        total_td2 = 0.0
        steps = 0
        done = False

        while not done and steps < 1000:
            action = self.epsilon_greedy(state)
            next_state, reward, done, _ = self.env.step(action)

            # TD-target et TD-error
            best_next = np.max(self.Q[next_state])
            td_target = reward + self.gamma * best_next
            td_error  = td_target - self.Q[state, action]

            # Mise à jour Q-Learning
            self.Q[state, action] += self.alpha * td_error

            total_reward += reward
            total_td2 += td_error ** 2
            state = next_state
            steps += 1

        avg_loss = total_td2 / steps if steps > 0 else 0.0
        return {'reward': total_reward, 'loss': avg_loss, 'steps': steps}

    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Entraîne sur num_episodes:
        - appel de train_episode()
        - décroissance linéaire de ε vers 0.01
        - extraction de la politique gloutonne finale
        """
        self.history.clear()
        decay = (self.initial_epsilon - 0.01) / num_episodes

        for ep in range(1, num_episodes + 1):
            stats = self.train_episode()

            # Décroissance d'epsilon
            self.epsilon = max(0.01, self.epsilon - decay)

            self.history.append({
                'episode': ep,
                'reward':  stats['reward'],
                'loss':    stats['loss'],
                'steps':   stats['steps'],
                'epsilon': self.epsilon
            })
            if ep % max(1, num_episodes // 10) == 0:
                print(f"[Q-Learning] Ep {ep}/{num_episodes}  "
                      f"R={stats['reward']:.2f}  loss={stats['loss']:.4f}  ε={self.epsilon:.3f}")

        # Construire la politique gloutonne à partir de Q
        for s in range(self.nS):
            self.policy[s] = int(np.argmax(self.Q[s]))

        return {'Q': self.Q, 'policy': self.policy, 'history': self.history}

    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Évalue la policy gloutonne (ε=0) sur num_episodes.
        """
        old_eps = self.epsilon
        self.epsilon = 0.0

        rewards, lengths = [], []
        for _ in range(num_episodes):
            state = self.env.reset()
            done, G, steps = False, 0.0, 0
            while not done and steps < 1000:
                a = self.policy[state]
                state, r, done, _ = self.env.step(a)
                G += r; steps += 1
            rewards.append(G); lengths.append(steps)

        self.epsilon = old_eps
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_steps':  np.mean(lengths)
        }

    def save(self, filepath: str):
        save_model({
            'Q':       self.Q,
            'policy':  self.policy,
            'alpha':   self.alpha,
            'gamma':   self.gamma,
            'epsilon': self.epsilon,
            'history': self.history
        }, filepath)

    def load(self, filepath: str):
        data = load_model(filepath)
        self.Q       = data['Q']
        self.policy  = data['policy']
        self.alpha   = data['alpha']
        self.gamma   = data['gamma']
        self.epsilon = data['epsilon']
        self.history = data['history']


class ExpectedSarsa:
    """
    Expected SARSA – On-policy TD Control using the expectation under ε-greedy.

    Met à jour Q(s,a) selon :
      Q(s,a) ← Q(s,a) + α [ r + γ 𝔼_{a'∼π}[Q(s',a')] − Q(s,a) ]
    """

    def __init__(
        self,
        env: Any,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
    ):
        """
        Args:
            env: Environnement Gym-like
            alpha: pas d'apprentissage ∈ (0,1]
            gamma: facteur d'actualisation ∈ [0,1]
            epsilon: ε initial pour ε-greedy
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon

        # Nombre d'états/actions
        if hasattr(env, 'nS') and hasattr(env, 'nA'):
            self.nS, self.nA = env.nS, env.nA
        else:
            self.nS = env.observation_space.n
            self.nA = env.action_space.n

        # Table Q(s,a)
        self.Q = np.zeros((self.nS, self.nA), dtype=float)
        # Politique gloutonne extraite après training
        self.policy = np.zeros(self.nS, dtype=int)

        # Historique des métriques
        self.history: List[Dict[str, float]] = []

    def epsilon_greedy(self, state: int) -> int:
        """ε-greedy avec tie-breaking aléatoire."""
        if random.random() < self.epsilon:
            return random.randrange(self.nA)
        q_s = self.Q[state]
        best = np.flatnonzero(q_s == q_s.max())
        return int(random.choice(best))

    def expected_q(self, state: int) -> float:
        """
        Calcule l'espérance E_{a∼π}[Q(state,a)] sous ε-greedy :
        π(a|s) = ε/|A|  sauf pour a* = argmax Q(s,a) où π(a*|s)=1−ε+ε/|A|.
        """
        q_s = self.Q[state]
        greedy = int(np.argmax(q_s))
        base = self.epsilon / self.nA
        exp_q = base * q_s.sum() + (1.0 - self.epsilon) * q_s[greedy]
        return float(exp_q)

    def train_episode(self) -> Dict[str, float]:
        """
        Un épisode Expected SARSA :
        s ← env.reset()
        tant que s non terminal :
          a ← ε-greedy(s)
          r, s' ← env.step(a)
          G = expected_q(s')
          td = r + γ G − Q(s,a)
          Q(s,a) += α td
          s ← s'
        """
        state = self.env.reset()
        total_reward = 0.0
        total_td2 = 0.0
        steps = 0
        done = False

        while not done and steps < 1000:
            action = self.epsilon_greedy(state)
            next_state, reward, done, _ = self.env.step(action)

            # calcul de l'espérance
            exp_q_next = 0.0 if done else self.expected_q(next_state)
            td_target = reward + self.gamma * exp_q_next
            td_error  = td_target - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error

            total_reward += reward
            total_td2 += td_error ** 2
            state = next_state
            steps += 1

        avg_loss = total_td2 / steps if steps > 0 else 0.0
        return {'reward': total_reward, 'loss': avg_loss, 'steps': steps}

    def train(self, num_episodes: int = 1000) -> Dict[str, List[float]]:
        """
        Entraîne Expected SARSA sur num_episodes :
        - appel répétitif de train_episode()
        - décroissance linéaire de ε vers 0.01
        - extraction de la politique gloutonne finale
        """
        self.history.clear()
        decay = (self.initial_epsilon - 0.01) / num_episodes

        for ep in range(1, num_episodes + 1):
            stats = self.train_episode()
            self.epsilon = max(0.01, self.epsilon - decay)

            self.history.append({
                'episode':  ep,
                'reward':   stats['reward'],
                'loss':     stats['loss'],
                'steps':    stats['steps'],
                'epsilon':  self.epsilon
            })
            if ep % max(1, num_episodes//10) == 0:
                print(f"[ExpSARSA] Ep{ep}/{num_episodes}  R={stats['reward']:.2f}  loss={stats['loss']:.4f}  ε={self.epsilon:.3f}")

        # Politique gloutonne finale
        for s in range(self.nS):
            self.policy[s] = int(np.argmax(self.Q[s]))
        return {'Q': self.Q, 'policy': self.policy, 'history': self.history}

    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """Évalue la politique gloutonne (ε=0)."""
        old_eps = self.epsilon
        self.epsilon = 0.0

        rewards, lengths = [], []
        for _ in range(num_episodes):
            state = self.env.reset()
            done, G, steps = False, 0.0, 0
            while not done and steps < 1000:
                a = self.policy[state]
                state, r, done, _ = self.env.step(a)
                G += r; steps += 1
            rewards.append(G); lengths.append(steps)

        self.epsilon = old_eps
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_steps':  np.mean(lengths)
        }

    def save(self, filepath: str):
        save_model({
            'Q':       self.Q,
            'policy':  self.policy,
            'alpha':   self.alpha,
            'gamma':   self.gamma,
            'epsilon': self.epsilon,
            'history': self.history
        }, filepath)

    def load(self, filepath: str):
        data = load_model(filepath)
        self.Q       = data['Q']
        self.policy  = data['policy']
        self.alpha   = data['alpha']
        self.gamma   = data['gamma']
        self.epsilon = data['epsilon']
        self.history = data['history']