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
    Implémentation de l'algorithme Dyna-Q.
    
    Dyna-Q combine Q-Learning avec la planification en utilisant
    un modèle appris de l'environnement pour générer des expériences
    simulées supplémentaires.
    """
    
    def __init__(self, env: Any, alpha: float = 0.1, gamma: float = 0.95, 
                 epsilon: float = 0.1, n_planning: int = 5):
        """
        Initialise Dyna-Q.
        
        Args:
            env: Environnement
            alpha: Taux d'apprentissage
            gamma: Facteur de discount
            epsilon: Paramètre epsilon-greedy
            n_planning: Nombre d'étapes de planification par mise à jour
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning
        
        # Détection du type d'environnement
        if hasattr(env, 'observation_space') and hasattr(env, 'action_space'):
            # Environnement Gym
            self.nS = env.observation_space.n
            self.nA = env.action_space.n
        else:
            # Environnement personnalisé
            self.nS = getattr(env, 'nS', getattr(env, 'length', 100))
            self.nA = getattr(env, 'nA', 2)
        
        # Table Q et modèle de l'environnement
        self.Q = np.zeros((self.nS, self.nA))
        self.model = {}  # model[(state, action)] = (next_state, reward)
        self.visited_states = set()
        self.state_action_pairs = []
        
        self.policy = None
        self.history = []
        
    def epsilon_greedy_policy(self, state: int) -> int:
        """
        Politique epsilon-greedy.
        
        Args:
            state: État actuel
            
        Returns:
            Action sélectionnée
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.nA - 1)
        else:
            return int(np.argmax(self.Q[state]))
        
    def update_model(self, state: int, action: int, next_state: int, reward: float):
        """
        Met à jour le modèle de l'environnement.
        
        Args:
            state: État actuel
            action: Action prise
            next_state: État suivant observé
            reward: Récompense observée
        """
        self.model[(state, action)] = (reward, next_state)
        self.visited_states.add(state)
        
        # Maintenir la liste des paires état-action pour la planification
        if (state, action) not in self.state_action_pairs:
            self.state_action_pairs.append((state, action))
        
    def update_q_value(self, state: int, action: int, reward: float, next_state: int):
        """
        Met à jour la valeur Q selon l'équation Q-Learning.
        
        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
        """
        # Q-learning update
        target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
        
    def planning_step(self):
        """
        Effectue une étape de planification en utilisant le modèle.
        """
        if not self.state_action_pairs:
            return
            
        # Sélectionner aléatoirement une paire (état, action) déjà visitée
        state, action = random.choice(self.state_action_pairs)
        
        # Récupérer la récompense et l'état suivant du modèle
        reward, next_state = self.model[(state, action)]
        
        # Effectuer une mise à jour Q-learning simulée
        target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
        
    def train_step(self, state: int, action: int, reward: float, next_state: int):
        """
        Effectue une étape d'entraînement complète (apprentissage + planification).
        
        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
        """
        # Apprentissage direct : Q-learning sur l'expérience réelle
        self.update_q_value(state, action, reward, next_state)
        
        # Mise à jour du modèle
        self.update_model(state, action, next_state, reward)
        
        # Planification : n étapes de Q-learning simulé
        for _ in range(self.n_planning):
            self.planning_step()
        
    def train_episode(self) -> Dict[str, Any]:
        """
        Entraîne sur un épisode complet.
        
        Returns:
            Statistiques de l'épisode
        """
        if hasattr(self.env, 'reset'):
            # Environnement Gym
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]  # Nouveau format gym
        else:
            # Environnement personnalisé
            state = self.env.reset()
            
        total_reward = 0.0
        steps = 0
        
        while True:
            # Choisir une action selon epsilon-greedy
            action = self.epsilon_greedy_policy(state)
            
            # Prendre l'action
            if hasattr(self.env, 'step'):
                result = self.env.step(action)
                if len(result) == 4:
                    next_state, reward, done, info = result
                else:
                    next_state, reward, done, info, _ = result
            else:
                next_state, reward, done = self.env.step(action)
                
            # Entraîner (apprentissage direct + planification)
            self.train_step(state, action, reward, next_state)
            
            # Accumuler les statistiques
            total_reward += reward
            steps += 1
            
            # Préparer pour l'itération suivante
            state = next_state
            
            if done or steps >= 1000:  # Protection contre les boucles infinies
                break
        
        return {
            'episode_reward': total_reward,
            'episode_steps': steps,
            'epsilon': self.epsilon,
            'model_size': len(self.model)
        }
        
    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Dyna-Q.
        
        Args:
            num_episodes: Nombre d'épisodes d'entraînement
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        self.history = []
        
        # Variables pour EMA (Exponential Moving Average)
        ema_score = 0.0
        ema_count = 0
        
        all_ema_scores = []
        episode_rewards = []
        
        print(f"Démarrage de l'entraînement Dyna-Q pour {num_episodes} épisodes...")
        
        for episode in range(1, num_episodes + 1):
            # Entraîner sur un épisode
            episode_stats = self.train_episode()
            
            # Mettre à jour EMA
            ema_score = 0.95 * ema_score + (1 - 0.95) * episode_stats['episode_reward']
            ema_count += 1
            
            # Correction du biais EMA
            corrected_ema_score = ema_score / (1 - 0.95 ** ema_count)
            
            # Enregistrer les statistiques
            self.history.append({
                'episode': episode,
                'episode_reward': episode_stats['episode_reward'],
                'episode_steps': episode_stats['episode_steps'],
                'ema_score': corrected_ema_score,
                'epsilon': self.epsilon,
                'model_size': episode_stats['model_size']
            })
            
            # Stocker pour les graphiques
            all_ema_scores.append(corrected_ema_score)
            episode_rewards.append(episode_stats['episode_reward'])
            
            # Affichage périodique
            if episode % max(1, num_episodes // 10) == 0:
                print(f"Épisode {episode}: EMA Score={corrected_ema_score:.4f}, "
                      f"Modèle: {episode_stats['model_size']} paires état-action")
        
        # Extraire la politique finale
        self.policy = self.extract_policy()
        
        print(f"Entraînement terminé après {num_episodes} épisodes")
        print(f"Score EMA final: {corrected_ema_score:.4f}")
        print(f"Taille du modèle final: {len(self.model)} paires état-action")
        
        return {
            'all_ema_scores': all_ema_scores,
            'episode_rewards': episode_rewards,
            'final_policy': self.policy,
            'final_q_values': self.Q.copy(),
            'final_model': self.model.copy(),
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
            policy[s] = int(np.argmax(self.Q[s, :]))
        return policy
        
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        model_data = {
            'Q': self.Q.tolist(),
            'model': {str(k): v for k, v in self.model.items()},
            'policy': self.policy.tolist() if self.policy is not None else None,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'n_planning': self.n_planning,
            'history': self.history
        }
        save_model(model_data, filepath)
        
    def load(self, filepath: str):
        """Charge un modèle sauvegardé."""
        model_data = load_model(filepath)
        self.Q = np.array(model_data['Q'])
        self.model = {eval(k): v for k, v in model_data['model'].items()}
        self.policy = np.array(model_data['policy']) if model_data['policy'] else None
        self.alpha = model_data['alpha']
        self.gamma = model_data['gamma']
        self.epsilon = model_data['epsilon']
        self.n_planning = model_data['n_planning']
        self.history = model_data['history']


class DynaQPlus:
    """
    Implémentation de l'algorithme Dyna-Q+.
    
    Dyna-Q+ étend Dyna-Q en ajoutant un bonus d'exploration pour
    encourager l'exploration des états-actions qui n'ont pas été
    visités récemment.
    """
    
    def __init__(self, env: Any, alpha: float = 0.1, gamma: float = 0.95, 
                 epsilon: float = 0.1, n_planning: int = 5, kappa: float = 0.001):
        """
        Initialise Dyna-Q+.
        
        Args:
            env: Environnement
            alpha: Taux d'apprentissage
            gamma: Facteur de discount
            epsilon: Paramètre epsilon-greedy
            n_planning: Nombre d'étapes de planification par mise à jour
            kappa: Paramètre du bonus d'exploration
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning
        self.kappa = kappa  # Coefficient du bonus de découverte
        
        # Détection du type d'environnement
        if hasattr(env, 'observation_space') and hasattr(env, 'action_space'):
            # Environnement Gym
            self.nS = env.observation_space.n
            self.nA = env.action_space.n
        else:
            # Environnement personnalisé
            self.nS = getattr(env, 'nS', getattr(env, 'length', 100))
            self.nA = getattr(env, 'nA', 2)
        
        # Table Q et modèle de l'environnement
        self.Q = np.zeros((self.nS, self.nA))
        self.model = {}  # model[(state, action)] = (next_state, reward)
        self.last_visit = {}  # last_visit[(state, action)] = dernier pas de vraie expérience
        self.current_time = 0  # Compteur de pas globaux
        
        self.visited_states = set()
        self.state_action_pairs = []
        
        self.policy = None
        self.history = []
        
    def epsilon_greedy_policy(self, state: int) -> int:
        """
        Politique epsilon-greedy.
        
        Args:
            state: État actuel
            
        Returns:
            Action sélectionnée
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.nA - 1)
        else:
            return int(np.argmax(self.Q[state]))
        
    def update_model(self, state: int, action: int, next_state: int, reward: float):
        """
        Met à jour le modèle de l'environnement et les temps de visite.
        
        Args:
            state: État actuel
            action: Action prise
            next_state: État suivant observé
            reward: Récompense observée
        """
        # Incrémenter le temps et enregistrer la vraie expérience
        self.current_time += 1
        self.last_visit[(state, action)] = self.current_time
        
        # Mise à jour du modèle
        self.model[(state, action)] = (reward, next_state)
        self.visited_states.add(state)
        
        # Maintenir la liste des paires état-action pour la planification
        if (state, action) not in self.state_action_pairs:
            self.state_action_pairs.append((state, action))
        
    def exploration_bonus(self, state: int, action: int) -> float:
        """
        Calcule le bonus d'exploration pour un couple état-action.
        
        Args:
            state: État
            action: Action
            
        Returns:
            Bonus d'exploration
        """
        # Calcul du bonus : racine du temps écoulé depuis dernière vraie visite
        tau = self.current_time - self.last_visit.get((state, action), 0)
        return self.kappa * math.sqrt(tau)
        
    def update_q_value(self, state: int, action: int, reward: float, next_state: int):
        """
        Met à jour la valeur Q selon l'équation Q-Learning.
        
        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
        """
        # Q-learning update (sans bonus pour l'expérience réelle)
        target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
        
    def planning_step(self):
        """
        Effectue une étape de planification avec bonus d'exploration.
        """
        if not self.state_action_pairs:
            return
            
        # Sélectionner aléatoirement une paire (état, action) déjà visitée
        state, action = random.choice(self.state_action_pairs)
        
        # Récupérer la récompense et l'état suivant du modèle
        reward, next_state = self.model[(state, action)]
        
        # Calculer le bonus d'exploration
        bonus = self.exploration_bonus(state, action)
        
        # Effectuer une mise à jour Q-learning simulée avec bonus
        target = reward + bonus + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
        
    def train_step(self, state: int, action: int, reward: float, next_state: int):
        """
        Effectue une étape d'entraînement complète (apprentissage + planification).
        
        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
        """
        # Mise à jour du modèle et des temps de visite
        self.update_model(state, action, next_state, reward)
        
        # Apprentissage direct : Q-learning sur l'expérience réelle
        self.update_q_value(state, action, reward, next_state)
        
        # Planification : n étapes de Q-learning simulé avec bonus
        for _ in range(self.n_planning):
            self.planning_step()
        
    def train_episode(self) -> Dict[str, Any]:
        """
        Entraîne sur un épisode complet.
        
        Returns:
            Statistiques de l'épisode
        """
        if hasattr(self.env, 'reset'):
            # Environnement Gym
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]  # Nouveau format gym
        else:
            # Environnement personnalisé
            state = self.env.reset()
            
        total_reward = 0.0
        steps = 0
        
        while True:
            # Choisir une action selon epsilon-greedy
            action = self.epsilon_greedy_policy(state)
            
            # Prendre l'action
            if hasattr(self.env, 'step'):
                result = self.env.step(action)
                if len(result) == 4:
                    next_state, reward, done, info = result
                else:
                    next_state, reward, done, info, _ = result
            else:
                next_state, reward, done = self.env.step(action)
                
            # Entraîner (apprentissage direct + planification)
            self.train_step(state, action, reward, next_state)
            
            # Accumuler les statistiques
            total_reward += reward
            steps += 1
            
            # Préparer pour l'itération suivante
            state = next_state
            
            if done or steps >= 1000:  # Protection contre les boucles infinies
                break
        
        return {
            'episode_reward': total_reward,
            'episode_steps': steps,
            'epsilon': self.epsilon,
            'model_size': len(self.model),
            'current_time': self.current_time
        }
        
    def train(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Dyna-Q+.
        
        Args:
            num_episodes: Nombre d'épisodes d'entraînement
            
        Returns:
            Dictionnaire contenant les résultats d'entraînement
        """
        self.history = []
        
        # Variables pour EMA (Exponential Moving Average)
        ema_score = 0.0
        ema_count = 0
        
        all_ema_scores = []
        episode_rewards = []
        
        print(f"Démarrage de l'entraînement Dyna-Q+ pour {num_episodes} épisodes...")
        
        for episode in range(1, num_episodes + 1):
            # Entraîner sur un épisode
            episode_stats = self.train_episode()
            
            # Mettre à jour EMA
            ema_score = 0.95 * ema_score + (1 - 0.95) * episode_stats['episode_reward']
            ema_count += 1
            
            # Correction du biais EMA
            corrected_ema_score = ema_score / (1 - 0.95 ** ema_count)
            
            # Enregistrer les statistiques
            self.history.append({
                'episode': episode,
                'episode_reward': episode_stats['episode_reward'],
                'episode_steps': episode_stats['episode_steps'],
                'ema_score': corrected_ema_score,
                'epsilon': self.epsilon,
                'model_size': episode_stats['model_size'],
                'current_time': episode_stats['current_time']
            })
            
            # Stocker pour les graphiques
            all_ema_scores.append(corrected_ema_score)
            episode_rewards.append(episode_stats['episode_reward'])
            
            # Affichage périodique
            if episode % max(1, num_episodes // 10) == 0:
                print(f"Épisode {episode}: EMA Score={corrected_ema_score:.4f}, "
                      f"Modèle: {episode_stats['model_size']} paires état-action, "
                      f"Temps: {episode_stats['current_time']}")
        
        # Extraire la politique finale
        self.policy = self.extract_policy()
        
        print(f"Entraînement terminé après {num_episodes} épisodes")
        print(f"Score EMA final: {corrected_ema_score:.4f}")
        print(f"Taille du modèle final: {len(self.model)} paires état-action")
        print(f"Temps total: {self.current_time} pas")
        
        return {
            'all_ema_scores': all_ema_scores,
            'episode_rewards': episode_rewards,
            'final_policy': self.policy,
            'final_q_values': self.Q.copy(),
            'final_model': self.model.copy(),
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
            policy[s] = int(np.argmax(self.Q[s, :]))
        return policy
        
    def save(self, filepath: str):
        """Sauvegarde le modèle."""
        model_data = {
            'Q': self.Q.tolist(),
            'model': {str(k): v for k, v in self.model.items()},
            'last_visit': {str(k): v for k, v in self.last_visit.items()},
            'current_time': self.current_time,
            'policy': self.policy.tolist() if self.policy is not None else None,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'n_planning': self.n_planning,
            'kappa': self.kappa,
            'history': self.history
        }
        save_model(model_data, filepath)
        
    def load(self, filepath: str):
        """Charge un modèle sauvegardé."""
        model_data = load_model(filepath)
        self.Q = np.array(model_data['Q'])
        self.model = {eval(k): v for k, v in model_data['model'].items()}
        self.last_visit = {eval(k): v for k, v in model_data['last_visit'].items()}
        self.current_time = model_data['current_time']
        self.policy = np.array(model_data['policy']) if model_data['policy'] else None
        self.alpha = model_data['alpha']
        self.gamma = model_data['gamma']
        self.epsilon = model_data['epsilon']
        self.n_planning = model_data['n_planning']
        self.kappa = model_data['kappa']
        self.history = model_data['history'] 