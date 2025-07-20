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
        # Priorité 1: Environnements avec get_mdp_info (nos environnements personnalisés)
        if hasattr(self.env, 'get_mdp_info'):
            mdp_info = self.env.get_mdp_info()
            self.nS = len(mdp_info['states'])
            self.nA = len(mdp_info['actions'])
            self.transition_probs = mdp_info['transition_matrix']
            self.rewards = mdp_info['reward_matrix']
            self.terminal_states = mdp_info.get('terminals', [])
            
        # Priorité 2: Environnements Gym avec attribut P
        elif hasattr(self.env, 'P'):
            # Déterminer le nombre d'états et d'actions
            if hasattr(self.env, 'nS') and hasattr(self.env, 'nA'):
                self.nS = self.env.nS
                self.nA = self.env.nA
            else:
                # Environnement générique
                self.nS = getattr(self.env, 'observation_space', type('obj', (object,), {'n': 16})).n
                self.nA = getattr(self.env, 'action_space', type('obj', (object,), {'n': 4})).n
            
            # Extraire les probabilités de transition
            self.transition_probs = self._extract_transition_probabilities()
            self.rewards = self._extract_rewards()
            
            # Identifier les états terminaux
            self.terminal_states = []
            if hasattr(self.env, 'desc'):
                # Pour FrozenLake par exemple
                desc = self.env.desc
                for i in range(desc.shape[0]):
                    for j in range(desc.shape[1]):
                        if desc[i, j] in [b'H', b'G']:  # Hole ou Goal
                            self.terminal_states.append(i * desc.shape[1] + j)
        
        # Priorité 3: Environnements sans MDP explicite
        else:
            # Déterminer le nombre d'états et d'actions
            if hasattr(self.env, 'nS') and hasattr(self.env, 'nA'):
                self.nS = self.env.nS
                self.nA = self.env.nA
            else:
                # Environnement générique
                self.nS = getattr(self.env, 'observation_space', type('obj', (object,), {'n': 16})).n
                self.nA = getattr(self.env, 'action_space', type('obj', (object,), {'n': 4})).n
            
            self.transition_probs = None
            self.rewards = [0.0, 1.0]  # Récompenses par défaut
            self.terminal_states = []
        
        # Initialiser Q et politique
        self.Q = np.zeros((self.nS, self.nA), dtype=float)
        self.policy = np.zeros(self.nS, dtype=int)
        
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
            next_state, reward, done, _ = self.env.step(a)
            return next_state, reward
        
        # Utiliser les probabilités de transition
        num_states = self.transition_probs.shape[2]
        
        # Construire une liste de (s', reward) avec prob > 0
        candidates = []
        probs = []
        
        for s_p in range(num_states):
            prob = self.transition_probs[s, a, s_p]
            if prob > 0:
                # Pour les environnements avec get_mdp_info, la récompense est dans reward_matrix
                if hasattr(self.env, 'get_mdp_info'):
                    reward = self.rewards[s, a]  # reward_matrix[s, a]
                else:
                    # Pour les environnements Gym, utiliser la récompense associée à la transition
                    reward = self.rewards[0]  # Fallback
                
                candidates.append((s_p, reward))
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
            s_p, reward = candidates[idx]
            return s_p, reward
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
