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
    Monte Carlo Control with Exploring Starts (MC-ES).

    Garantit que chaque couple (état, action) a une probabilité non nulle d'être
    utilisé comme point de départ d'épisode.
    """

    def __init__(self, env: Any, gamma: float = 1.0):
        """
        Args:
            env: Environnement conforme à l'API OpenAI Gym
            gamma: facteur d'actualisation
        """
        self.env = env
        self.gamma = gamma

        # Structures MDP
        self._initialize_mdp_structures()

        # Historique d'entraînement
        self.history: List[Dict[str, float]] = []

    def _initialize_mdp_structures(self):
        # Nombre d'états et d'actions
        if hasattr(self.env, 'nS') and hasattr(self.env, 'nA'):
            self.nS, self.nA = self.env.nS, self.env.nA
        else:
            # Fallback pour Gym classique
            self.nS = getattr(self.env.observation_space, 'n', None)
            self.nA = getattr(self.env.action_space, 'n', None)
            if self.nS is None or self.nA is None:
                raise ValueError("Impossible de déterminer nS ou nA à partir de l'env.")
        
        # Initialisation de Q et de la politique
        self.Q = np.zeros((self.nS, self.nA), dtype=float)
        self.policy = np.zeros(self.nS, dtype=int)

        # Sommes et comptes des retours
        self.returns_sum = defaultdict(float)   # clé = (s,a)
        self.returns_count = defaultdict(int)

    def generate_episode(self, start_state: int, start_action: int) -> List[Tuple[int,int,float]]:
        """Génère un épisode en forçant (s₀, a₀) = (start_state, start_action)."""
        episode = []
        # Réinitialisation: récupérer l'état initial
        obs = self.env.reset()
        # Forcer l’état si possible (Gym supporte env.unwrapped.s = ...)
        try:
            self.env.unwrapped.state = start_state
            state = start_state
        except:
            state = obs  # on suppose alors que reset() renvoie directement l'état

        action = start_action
        done = False
        while not done:
            next_obs, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_obs
            action = self.policy[state]
        return episode

    def update_q_function(self, episode: List[Tuple[int,int,float]]):
        """First-visit MC: met à jour Q et les compteurs de retours."""
        G = 0.0
        visited = set()
        # On parcourt l’épisode à l’envers
        for (s, a, r) in reversed(episode):
            G = self.gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                self.returns_count[(s,a)] += 1
                self.returns_sum[(s,a)] += G
                self.Q[s,a] = self.returns_sum[(s,a)] / self.returns_count[(s,a)]

    def improve_policy(self):
        """Politique gloutonne basée sur Q avec tie-breaking aléatoire."""
        for s in range(self.nS):
            q_s = self.Q[s]
            # argmax avec random tie-break
            max_val = np.max(q_s)
            candidates = np.flatnonzero(q_s == max_val)
            self.policy[s] = random.choice(candidates)

    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """Boucle principale d'entraînement MC-ES."""
        # Politique initiale aléatoire
        for s in range(self.nS):
            self.policy[s] = random.randrange(self.nA)

        self.history.clear()
        for ep in range(1, num_episodes + 1):
            # Exploring starts
            s0 = random.randrange(self.nS)
            a0 = random.randrange(self.nA)

            episode = self.generate_episode(s0, a0)
            if not episode:
                continue

            self.update_q_function(episode)
            self.improve_policy()

            # Stats
            total_r = sum(r for _,_,r in episode)
            avg_q = np.mean(self.Q)
            self.history.append({
                'episode': ep,
                'reward': total_r,
                'avg_q': avg_q,
                'length': len(episode)
            })
            if ep % 100 == 0:
                print(f"[MC-ES] Épisode {ep}: Récompense={total_r:.2f}, Q_moy={avg_q:.4f}")

        return {
            'Q': self.Q,
            'policy': self.policy,
            'history': self.history
        }

    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """Évalue la politique apprise sur plusieurs épisodes."""
        if self.policy is None:
            raise RuntimeError("Politique non initialisée.")
        rewards, lengths = [], []
        succ = 0
        for _ in range(num_episodes):
            obs = self.env.reset()
            state = obs if not hasattr(self.env, 'state') else self.env.state
            done, G = False, 0.0
            steps = 0
            while not done:
                a = self.policy[state]
                state, r, done, _ = self.env.step(a)
                G += r; steps += 1
            rewards.append(G); lengths.append(steps)
            if G > 0: succ += 1

        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'success_rate': succ / num_episodes,
            'avg_length': np.mean(lengths)
        }

    def get_action(self, state: int) -> int:
        """Pour déploiement pas-à-pas."""
        return int(self.policy[state])

    def save(self, filepath: str):
        save_model({
            'Q': self.Q,
            'policy': self.policy,
            'returns_sum': dict(self.returns_sum),
            'returns_count': dict(self.returns_count),
            'gamma': self.gamma,
            'history': self.history
        }, filepath)

    def load(self, filepath: str):
        data = load_model(filepath)
        self.Q = data['Q']
        self.policy = data['policy']
        self.returns_sum = defaultdict(float, data['returns_sum'])
        self.returns_count = defaultdict(int, data['returns_count'])
        self.gamma = data['gamma']
        self.history = data['history']


class OnPolicyMC:
    """
    On-policy First-Visit Monte Carlo Control avec ε-greedy.

    L’algorithme apprend Q et simultanément une politique ε-gloutonne.
    """

    def __init__(self, env: Any, gamma: float = 1.0, epsilon: float = 0.1):
        """
        Args:
            env: Environnement conforme Gym
            gamma: fact. d’actualisation
            epsilon: paramètre pour l’exploration ε-greedy
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon

        # Structures MDP
        self._initialize_mdp_structures()

        # Historique d’entraînement
        self.history: List[Dict[str, float]] = []

    def _initialize_mdp_structures(self):
        """Détecte nS, nA puis initialise Q, policy, retours."""
        if hasattr(self.env, 'nS') and hasattr(self.env, 'nA'):
            self.nS, self.nA = self.env.nS, self.env.nA
        else:
            # Gym standard
            self.nS = getattr(self.env.observation_space, 'n', None)
            self.nA = getattr(self.env.action_space, 'n', None)
            if self.nS is None or self.nA is None:
                raise ValueError("Impossible de déterminer nS ou nA.")
        # Valeurs initiales
        self.Q = np.zeros((self.nS, self.nA), dtype=float)
        self.policy = np.zeros(self.nS, dtype=int)
        self.returns_sum   = defaultdict(float)  # clé = (s,a)
        self.returns_count = defaultdict(int)

    def epsilon_greedy_action(self, state: int) -> int:
        """Sélectionne a selon ε-greedy sur Q[state]."""
        if random.random() < self.epsilon:
            return random.randrange(self.nA)
        # tie-breaking aléatoire
        q_s = self.Q[state]
        max_val = np.max(q_s)
        best = np.flatnonzero(q_s == max_val)
        return int(random.choice(best))

    def generate_episode(self) -> List[Tuple[int,int,float]]:
        """
        Simule un épisode complet selon la politique ε-greedy.
        Retourne la liste [(s₀,a₀,r₁), (s₁,a₁,r₂), …].
        """
        episode = []
        obs = self.env.reset()
        state = obs if not hasattr(self.env, 'state') else self.env.state

        done, steps = False, 0
        while not done and steps < 1000:
            action = self.epsilon_greedy_action(state)
            next_obs, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_obs
            steps += 1

        return episode

    def update_q_function(self, episode: List[Tuple[int,int,float]]):
        """
        First-visit MC : pour chaque (s,a) dans l’épisode
        on calcule le retour G et on met à jour Q(s,a).
        """
        G = 0.0
        seen = set()
        # on parcourt à l’envers
        for s, a, r in reversed(episode):
            G = self.gamma * G + r
            if (s,a) not in seen:
                seen.add((s,a))
                self.returns_count[(s,a)] += 1
                self.returns_sum[(s,a)]   += G
                self.Q[s,a] = self.returns_sum[(s,a)] / self.returns_count[(s,a)]

    def improve_policy(self):
        """Rend la policy gloutonne par rapport à Q (tie-break aléatoire)."""
        for s in range(self.nS):
            q_s = self.Q[s]
            max_val = np.max(q_s)
            best = np.flatnonzero(q_s == max_val)
            self.policy[s] = int(random.choice(best))

    def train(self, num_episodes: int = 1000) -> Dict[str,Any]:
        """
        Boucle principale :
        - génère épisode par ε-greedy
        - met à jour Q
        - améliore π
        - decaye ε
        """
        self.history.clear()
        # initialisation key-in-hand: policy aléatoire
        for s in range(self.nS):
            self.policy[s] = random.randrange(self.nA)

        for ep in range(1, num_episodes+1):
            episode = self.generate_episode()
            if not episode:
                continue

            self.update_q_function(episode)
            self.improve_policy()
            # decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)

            # stats
            G = sum(r for _,_,r in episode)
            avg_q = float(np.mean(self.Q))
            self.history.append({
                'episode': ep,
                'reward': G,
                'avg_q': avg_q,
                'epsilon': self.epsilon,
                'length': len(episode)
            })
            if ep % 100 == 0:
                print(f"[OnPolicyMC] Ép. {ep}: R={G:.2f}, ε={self.epsilon:.3f}")

        return {
            'Q': self.Q,
            'policy': self.policy,
            'history': self.history
        }

    def evaluate(self, num_episodes: int = 100) -> Dict[str,float]:
        """Évalue la politique apprise sans exploration (ε=0)."""
        rewards, lengths, succ = [], [], 0
        # forcer greedy
        old_eps = self.epsilon
        self.epsilon = 0.0

        for _ in range(num_episodes):
            obs = self.env.reset()
            state = obs if not hasattr(self.env, 'state') else self.env.state
            done, G, steps = False, 0.0, 0
            while not done and steps < 1000:
                a = self.policy[state]
                state, r, done, _ = self.env.step(a)
                G += r; steps += 1
            rewards.append(G); lengths.append(steps)
            if G > 0: succ += 1

        self.epsilon = old_eps
        return {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'success_rate': succ/num_episodes,
            'avg_length': np.mean(lengths)
        }

    def get_action(self, state: int) -> int:
        """Pour déploiement pas-à-pas : action gloutonne."""
        return int(self.policy[state])

    def save(self, filepath: str):
        save_model({
            'Q': self.Q,
            'policy': self.policy,
            'returns_sum': dict(self.returns_sum),
            'returns_count': dict(self.returns_count),
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'history': self.history
        }, filepath)

    def load(self, filepath: str):
        data = load_model(filepath)
        self.Q             = data['Q']
        self.policy        = data['policy']
        self.returns_sum   = defaultdict(float, data['returns_sum'])
        self.returns_count = defaultdict(int,   data['returns_count'])
        self.gamma         = data['gamma']
        self.epsilon       = data['epsilon']
        self.history       = data['history']


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