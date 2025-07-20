"""
Module de Planning pour le Reinforcement Learning.

Ce module implémente les algorithmes de planification intégrée :
- Dyna-Q (Intégration d'apprentissage direct et de planification)
- Dyna-Q+ (Extension de Dyna-Q avec bonus d'exploration)

Ces algorithmes combinent l'apprentissage par renforcement avec
la planification basée sur un modèle appris de l'environnement.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from collections import defaultdict
import random
import math
from src.utils_io import save_model, load_model


class DynaQ:
    """
    Dyna-Q (Sutton & Barto, Alg. 8.3).

    Combinaison de Q-Learning off-policy et de planification via un modèle
    de transitions apprise en ligne (one‐step model).
    """

    def __init__(
        self,
        env: Any,
        alpha: float      = 0.1,
        gamma: float      = 0.95,
        epsilon: float    = 0.1,
        planning_steps: int = 5
    ):
        """
        Args:
          env           : Environnement Gym-like
          alpha         : taux d'apprentissage α
          gamma         : facteur d'actualisation γ
          epsilon       : paramètre ε pour ε-greedy
          planning_steps: nombre d'updates de planification par pas réel
        """
        self.env           = env
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.initial_epsilon = epsilon
        self.planning_steps = planning_steps

        # Détection de nS et nA
        if hasattr(env, 'nS') and hasattr(env, 'nA'):
            self.nS, self.nA = env.nS, env.nA
        else:
            self.nS = env.observation_space.n
            self.nA = env.action_space.n

        # Table Q et modèle one-step
        self.Q     = np.zeros((self.nS, self.nA), dtype=float)
        self.model: Dict[Tuple[int,int], Tuple[int,float]] = {}
        self.sa_pairs: List[Tuple[int,int]] = []

        # Pour le suivi
        self.policy: np.ndarray = np.zeros(self.nS, dtype=int)
        self.history: List[Dict[str, float]] = []

    def epsilon_greedy(self, s: int) -> int:
        """ε-greedy avec tie-breaking aléatoire."""
        if random.random() < self.epsilon:
            return random.randrange(self.nA)
        q_s = self.Q[s]
        best = np.flatnonzero(q_s == q_s.max())
        return int(random.choice(best))

    def q_update(self, s: int, a: int, r: float, s2: int):
        """
        Q-Learning update (experience réelle ou simulée) :
          Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s2,a') − Q(s,a) ].
        """
        target = r + self.gamma * np.max(self.Q[s2])
        self.Q[s,a] += self.alpha * (target - self.Q[s,a])

    def update_model(self, s: int, a: int, r: float, s2: int):
        """
        Stocke la dernière transition pour (s,a).
        One-step model : (s,a) → (s2,r).
        """
        if (s,a) not in self.model:
            self.sa_pairs.append((s,a))
        self.model[(s,a)] = (s2, r)

    def planning(self):
        """Effectue plusieurs itérations de Q-update simulées via le modèle."""
        for _ in range(self.planning_steps):
            s,a = random.choice(self.sa_pairs)
            s2,r = self.model[(s,a)]
            self.q_update(s, a, r, s2)

    def train_episode(self) -> Dict[str,float]:
        """
        Un épisode complet de Dyna-Q :
        - interaction réelle
        - Q-update et model-update
        - phase de planning
        """
        s = self.env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < 1000:
            a = self.epsilon_greedy(s)
            s2, r, done, _ = self.env.step(a)

            # 1) apprentissage réel
            self.q_update(s, a, r, s2)
            # 2) mise à jour du modèle
            self.update_model(s, a, r, s2)
            # 3) planification
            self.planning()

            total_reward += r
            s = s2
            steps += 1

        return {'reward': total_reward, 'steps': steps}

    def train(self, num_episodes: int = 1000) -> Dict[str,Any]:
        """
        Boucle d'entraînement complète :
        - num_episodes épisodes Dyna-Q
        - décroissance linéaire de ε vers 0.01
        - extraction de la politique gloutonne finale
        """
        decay = (self.initial_epsilon - 0.01) / num_episodes
        self.history.clear()

        for ep in range(1, num_episodes+1):
            stats = self.train_episode()
            # decay ε
            self.epsilon = max(0.01, self.epsilon - decay)

            self.history.append({
                'episode': ep,
                'reward':  stats['reward'],
                'steps':   stats['steps'],
                'epsilon': self.epsilon,
                'model_size': len(self.model)
            })
            if ep % max(1, num_episodes//10) == 0:
                print(f"[Dyna-Q] Ep{ep}/{num_episodes}  R={stats['reward']:.2f}  ε={self.epsilon:.3f}  Model={(len(self.model))}")

        # politique gloutonne finale
        for s in range(self.nS):
            self.policy[s] = int(np.argmax(self.Q[s]))

        return {
            'Q':          self.Q,
            'policy':     self.policy,
            'history':    self.history,
            'model':      self.model
        }

    def evaluate(self, num_episodes: int = 100) -> Dict[str,float]:
        """Évalue la politique gloutonne sans exploration."""
        old_eps = self.epsilon
        self.epsilon = 0.0
        rewards, lengths = [], []

        for _ in range(num_episodes):
            s = self.env.reset()
            done, G, steps = False, 0.0, 0
            while not done and steps < 1000:
                a = self.policy[s]
                s, r, done, _ = self.env.step(a)
                G += r; steps += 1
            rewards.append(G); lengths.append(steps)

        self.epsilon = old_eps
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_steps':  np.mean(lengths)
        }

    def save(self, path: str):
        save_model({
            'Q':            self.Q.tolist(),
            'policy':       self.policy.tolist(),
            'model':        {str(k): v for k,v in self.model.items()},
            'alpha':        self.alpha,
            'gamma':        self.gamma,
            'epsilon':      self.epsilon,
            'planning_steps': self.planning_steps,
            'history':      self.history
        }, path)

    def load(self, path: str):
        data = load_model(path)
        self.Q             = np.array(data['Q'])
        self.policy        = np.array(data['policy'])
        self.model         = {eval(k): tuple(v) for k,v in data['model'].items()}
        self.alpha         = data['alpha']
        self.gamma         = data['gamma']
        self.epsilon       = data['epsilon']
        self.planning_steps = data['planning_steps']
        self.history       = data['history']


class DynaQPlus:
    """
    Dyna-Q+ (Sutton & Barto, Alg. 8.4).

    Comme Dyna-Q, mais ajoute un bonus d'exploration κ·√τ pour chaque
    paire (s,a) non visitée récemment, où τ est le temps écoulé.
    """

    def __init__(
        self,
        env: Any,
        alpha: float         = 0.1,
        gamma: float         = 0.95,
        epsilon: float       = 0.1,
        planning_steps: int  = 5,
        kappa: float         = 0.001
    ):
        """
        Args:
            env           : Environnement Gym-like
            alpha         : taux d'apprentissage α
            gamma         : facteur d'actualisation γ
            epsilon       : paramètre ε pour ε-greedy
            planning_steps: nombre de pas de planification par interaction réelle
            kappa         : coefficient du bonus d'exploration
        """
        self.env            = env
        self.alpha          = alpha
        self.gamma          = gamma
        self.epsilon        = epsilon
        self.initial_epsilon = epsilon
        self.planning_steps = planning_steps
        self.kappa          = kappa

        # Détection de nS et nA
        if hasattr(env, 'nS') and hasattr(env, 'nA'):
            self.nS, self.nA = env.nS, env.nA
        else:
            self.nS = env.observation_space.n
            self.nA = env.action_space.n

        # Table Q et modèle one-step
        self.Q              = np.zeros((self.nS, self.nA), dtype=float)
        self.model: Dict[Tuple[int,int], Tuple[int,float]] = {}
        self.sa_pairs: List[Tuple[int,int]]             = []
        self.last_visit: Dict[Tuple[int,int], int]      = {}
        self.time                                         = 0

        # Pour le suivi
        self.policy = np.zeros(self.nS, dtype=int)
        self.history: List[Dict[str, float]] = []

    def epsilon_greedy(self, s: int) -> int:
        """ε-greedy avec tie-breaking aléatoire."""
        if random.random() < self.epsilon:
            return random.randrange(self.nA)
        q_s = self.Q[s]
        best = np.flatnonzero(q_s == q_s.max())
        return int(random.choice(best))

    def _update_q(self, s: int, a: int, r: float, s2: int, bonus: float = 0.0):
        """
        Q-learning update (réel ou simulé) avec eventuel bonus :
          Q(s,a) ← Q(s,a) + α [ r + bonus + γ max_a' Q(s2,a') − Q(s,a) ]
        """
        target = r + bonus + self.gamma * np.max(self.Q[s2])
        self.Q[s,a] += self.alpha * (target - self.Q[s,a])

    def _update_model(self, s: int, a: int, r: float, s2: int):
        """
        Stocke la transition observée et date de visite.
        """
        self.time += 1
        self.model[(s,a)]     = (s2, r)
        self.last_visit[(s,a)] = self.time
        if (s,a) not in self.sa_pairs:
            self.sa_pairs.append((s,a))

    def _planning(self):
        """Effectue des Q‐updates simulés via le modèle avec bonus."""
        for _ in range(self.planning_steps):
            s,a = random.choice(self.sa_pairs)
            s2,r = self.model[(s,a)]
            tau   = self.time - self.last_visit.get((s,a), 0)
            bonus = self.kappa * math.sqrt(tau)
            self._update_q(s, a, r, s2, bonus)

    def train_episode(self) -> Dict[str, float]:
        """
        Un épisode Dyna-Q+ :
        1) interaction réelle: Q-update + model-update
        2) planification
        """
        s = self.env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < 1000:
            a = self.epsilon_greedy(s)
            s2, r, done, _ = self.env.step(a)

            # 1) apprentissage réel
            self._update_q(s, a, r, s2)
            self._update_model(s, a, r, s2)
            # 2) planification
            self._planning()

            total_reward += r
            s = s2
            steps += 1

        return {'reward': total_reward, 'steps': steps, 'time': self.time}

    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Boucle d'entraînement Dyna-Q+ :
        - num_episodes épisodes
        - décroissance linéaire de ε vers 0.01
        - extraction de la politique gloutonne finale
        """
        decay = (self.initial_epsilon - 0.01) / num_episodes
        self.history.clear()

        for ep in range(1, num_episodes+1):
            stats = self.train_episode()
            self.epsilon = max(0.01, self.epsilon - decay)

            self.history.append({
                'episode': ep,
                'reward':  stats['reward'],
                'steps':   stats['steps'],
                'time':    stats['time'],
                'epsilon': self.epsilon,
                'model_size': len(self.model)
            })
            if ep % max(1, num_episodes//10) == 0:
                print(f"[Dyna-Q+] Ep{ep}/{num_episodes}  R={stats['reward']:.2f}  ε={self.epsilon:.3f}  τ={stats['time']}  |model|={len(self.model)}")

        # Politique finale gloutonne
        for s in range(self.nS):
            self.policy[s] = int(np.argmax(self.Q[s]))

        return {
            'Q':       self.Q,
            'policy':  self.policy,
            'history': self.history,
            'model':   self.model
        }

    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """Évalue la politique gloutonne (ε=0)."""
        old_eps = self.epsilon
        self.epsilon = 0.0
        rewards, lengths = [], []

        for _ in range(num_episodes):
            s = self.env.reset()
            done, G, steps = False, 0.0, 0
            while not done and steps < 1000:
                a = self.policy[s]
                s, r, done, _ = self.env.step(a)
                G += r; steps += 1
            rewards.append(G); lengths.append(steps)

        self.epsilon = old_eps
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_steps':  np.mean(lengths)
        }

    def save(self, path: str):
        save_model({
            'Q':         self.Q.tolist(),
            'policy':    self.policy.tolist(),
            'model':     {str(k): v for k,v in self.model.items()},
            'last_visit':{str(k): v for k,v in self.last_visit.items()},
            'time':      self.time,
            'alpha':     self.alpha,
            'gamma':     self.gamma,
            'epsilon':   self.epsilon,
            'planning_steps': self.planning_steps,
            'kappa':     self.kappa,
            'history':   self.history
        }, path)

    def load(self, path: str):
        data = load_model(path)
        self.Q             = np.array(data['Q'])
        self.policy        = np.array(data['policy'])
        self.model         = {eval(k): tuple(v) for k,v in data['model'].items()}
        self.last_visit    = {eval(k): v     for k,v in data['last_visit'].items()}
        self.time          = data['time']
        self.alpha         = data['alpha']
        self.gamma         = data['gamma']
        self.epsilon       = data['epsilon']
        self.planning_steps= data['planning_steps']
        self.kappa         = data['kappa']
        self.history       = data['history']