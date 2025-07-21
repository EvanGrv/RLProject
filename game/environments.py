"""
Environnements de jeux pour le reinforcement learning.
Extracté du notebook Environnement_RL.ipynb
"""

# Importation des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Optional, Any
import random
import math
from itertools import combinations


class BaseEnvironment(ABC):
    """Classe de base abstraite pour tous les environnements"""
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self, action, is_dynamic_programming=False):
        pass
    
    @property
    @abstractmethod
    def _build_transition_matrix(self):
        pass
    
    @property
    @abstractmethod
    def _build_reward_matrix(self):
        pass



# Line World
# Line est un environnement où l'agent doit se déplacer sur une ligne et collecter des récompenses.
# L'agent peut se déplacer vers la gauche ou la droite.
# L'agent reçoit une récompense de -1.0 si il se déplace vers la gauche 
# L'agent recoit une récompense de 1.0 si il se déplace vers la droite.
# L'agent reçoit une récompense de 0.0 si il se déplace vers le milieu de la ligne

# ENVIRONNEMENT LINE WORLD

class LineWorld(BaseEnvironment):
    def __init__(self, length=8):
        self.length = length
        self.start = length // 2
        self.terminals = [0, length-1]
        self.actions = [0, 1]
        self.reset()

        # Ajout pour compatibilité Gym
        self.observation_space = type('obj', (), {'n': self.length})()
        self.action_space = type('obj', (), {'n': len(self.actions)})()

        # Matrice de transition et récompenses pour les algorithmes de programmation dynamique
        self._transition_matrix = self._build_transition_matrix()
        self._reward_matrix = self._build_reward_matrix()

        # Métriques
        self.episode_count = 0
        self.steps_count = 0
        self.total_reward = 0
        self.visited_states = set()
        self.episode_history = []

    def reset(self):
        # Réinitialisation de l'environnement
        self.state = self.start
        return self.state

    def step(self, action, is_dynamic_programming=False):
        current_state = self.state
    
        # Calcul du prochain état
        if action == 0:  # Gauche
            reward = -1.0
            next_state = max(0, self.state - 1)
        else:  # Droite
            reward = 1.0
            next_state = min(self.length-1, self.state + 1)

        # Vérification du milieu APRÈS le calcul du next_state
        if next_state == self.length // 2:
            reward = 0.0

        done = next_state in self.terminals
        self.state = next_state

        if is_dynamic_programming:
            return {
                'state': self.state,
                'previous_state': current_state,
                'action': action
            }, reward, done
        else:
            return next_state, reward, done, {
                'action_name': 'gauche' if action == 0 else 'droite',
                'position': self.state,
                'previous_position': current_state
            }

    def visualisation(self):
        # Visualisation de notre environnement
        print("\n" + "="*40)
        line = ["[ ]"] * self.length
        line[self.state] = "[A]"  # Position de l'agent
        line[0] = "[T]"  # Terminal gauche
        line[-1] = "[T]"  # Terminal droit
        print(" ".join(line))
        
        # On affiche les indices pour mieux comprendre
        indices = [f"{i:^3}" for i in range(self.length)]
        print(" ".join(indices))
        print("="*40)

    def _build_transition_matrix(self):
        # Construit la matrice de transition P(s'|s,a).
        P = np.zeros((self.length, len(self.actions), self.length))
        
        for s in range(self.length):
            # Action gauche (0)
            next_s = max(0, s - 1)
            P[s, 0, next_s] = 1.0
            
            # Action droite (1)
            next_s = min(self.length-1, s+1)
            P[s, 1, next_s] = 1.0
            
        return P

    def _build_reward_matrix(self):
        # Construit la matrice de récompense R(s,a).
        R = np.zeros((self.length, len(self.actions)))
        
        for s in range(self.length):
            # Action gauche
            next_s_left = max(0, s-1)
            if next_s_left in self.terminals:
                R[s, 0] = 1.0

            # Action droite (1)
            next_s_right = min(self.length-1, s+1)
            if next_s_right in self.terminals:
                R[s, 1] = 1.0
                
        return R

    def get_mdp_info(self):
        # Retourne les informations du MDP
        return {
            'states': range(self.length),
            'actions': self.actions,
            'transition_matrix': self._transition_matrix,
            'reward_matrix': self._reward_matrix,
            'terminals': self.terminals,
            'gamma': 0.99  
        }

    def get_state_space(self):
        return range(self.length)

    def get_action_space(self):
        return self.actions

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property
    def reward_matrix(self):
        return self._reward_matrix



# Grid world
# Le Grid world est un environnement où l'agent doit se déplacer sur une grille et collecter des récompenses.
# L'agent peut se déplacer vers le haut, le bas, la gauche ou la droite.
# L'agent reçoit une récompense de -1.0 si il se déplace vers le haut, le bas, la gauche ou la droite.
# L'agent reçoit une récompense de 0.0 si il se déplace vers le milieu de la grille.

class GridWorld(BaseEnvironment):
    def __init__(self, n_rows=5, n_cols=5):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_states = n_rows * n_cols

        # Métriques sup
        self.steps_count = 0
        self.total_reward = 0
        self.visited_states = set()
        
        # Définition des actions : 0: Gauche, 1: Droite, 2: Haut, 3: Bas
        self.actions = [0, 1, 2, 3]
        
        # Les états terminaux sont dans les coins supérieur gauche et inférieur droit de la grille
        self.terminals = [0, self.n_states - 1]
        
        # CORRECTION: Récompenses équilibrées pour favoriser l'atteinte des terminaux
        self.rewards = {
            'movement': -0.1,  # Déplacement normal (petit coût pour favoriser l'efficacité)
            'middle': -1.0,    # Vers le milieu (pénalité modérée)
            'no_move': -0.5,   # Pas de mouvement possible
            'terminal': 10.0   # AJOUT: Récompense positive importante pour atteindre un terminal
        }
        
        # Position initiale (centre de la grille)
        self.start_row = self.n_rows // 2
        self.start_col = self.n_cols // 2
        self.start_state = self._pos_to_state(self.start_row, self.start_col)
        
        # Spécifique à la programmation dynamique
        self._transition_matrix = self._build_transition_matrix()
        self._reward_matrix = self._build_reward_matrix()
        
        # Initialisation
        self.reset()

    def reset(self):
        # Réinitialisation de l'environnement
        self.current_state = self.start_state
        self.done = False
        return self.current_state

    def step(self, action, for_dp=False):
        # Cette méthode est utilisée pour exécuter une action et retourner le résultat
        current_row, current_col = self._state_to_pos(self.current_state)
        new_row, new_col = current_row, current_col
        
        # Calcul de la nouvelle position
        if action == 0 and current_col > 0:  # Gauche
            new_col -= 1
        elif action == 1 and current_col < self.n_cols - 1:  # Droite
            new_col += 1
        elif action == 2 and current_row > 0:  # Haut
            new_row -= 1
        elif action == 3 and current_row < self.n_rows - 1:  # Bas
            new_row += 1
            
        next_state = self._pos_to_state(new_row, new_col)
        
        # CORRECTION: Calcul de la récompense avec bonus terminal
        if next_state in self.terminals:
            reward = self.rewards['terminal']  # Grande récompense pour atteindre un terminal
        elif (new_row, new_col) == (current_row, current_col):
            reward = self.rewards['no_move']
        elif next_state == self.start_state:
            reward = self.rewards['middle']
        else:
            reward = self.rewards['movement']
            
        self.done = next_state in self.terminals
        self.current_state = next_state
        
        # Mise à jour des métriques
        self.steps_count += 1
        self.total_reward += reward
        self.visited_states.add(next_state)

        if for_dp:
            return {
                'state': self.current_state,
                'position': (new_row, new_col),
                'previous_position': (current_row, current_col)
            }, reward, self.done
        else:
            return next_state, reward, self.done, {
                'position': (new_row, new_col),
                'action_name': ['gauche', 'droite', 'haut', 'bas'][action]
            }

    def _pos_to_state(self, row, col):
        # Convertit une position (row, col) en numéro d'état
        return row * self.n_cols + col

    def _state_to_pos(self, state):
        # Convertit un numéro d'état en position (row, col)
        return divmod(state, self.n_cols)

    def _build_transition_matrix(self):
        # Construit la matrice de transition P(s'|s,a)
        P = np.zeros((self.n_states, len(self.actions), self.n_states))
        
        for s in range(self.n_states):
            if s in self.terminals:
                # CORRECTION: Les états terminaux restent terminaux
                P[s, :, s] = 1.0
                continue
                
            row, col = self._state_to_pos(s)
            
            for a in self.actions:
                new_row, new_col = row, col
                
                if a == 0 and col > 0:
                    new_col -= 1
                elif a == 1 and col < self.n_cols - 1:
                    new_col += 1
                elif a == 2 and row > 0:
                    new_row -= 1
                elif a == 3 and row < self.n_rows - 1:
                    new_row += 1
                    
                next_state = self._pos_to_state(new_row, new_col)
                P[s, a, next_state] = 1.0
                
        return P

    def _build_reward_matrix(self):
        # CORRECTION: Construit la matrice des récompenses avec bonus terminal
        R = np.zeros((self.n_states, len(self.actions)))
        
        for s in range(self.n_states):
            if s in self.terminals:
                # Les états terminaux n'ont pas de récompenses d'actions
                continue
                
            row, col = self._state_to_pos(s)
            
            for a in self.actions:
                new_row, new_col = row, col
                
                if a == 0 and col > 0:
                    new_col -= 1
                elif a == 1 and col < self.n_cols - 1:
                    new_col += 1
                elif a == 2 and row > 0:
                    new_row -= 1
                elif a == 3 and row < self.n_rows - 1:
                    new_row += 1
                    
                next_state = self._pos_to_state(new_row, new_col)
                
                # CORRECTION: Récompense positive pour atteindre les terminaux
                if next_state in self.terminals:
                    R[s, a] = self.rewards['terminal']
                elif (new_row, new_col) == (row, col):
                    R[s, a] = self.rewards['no_move']
                elif next_state == self.start_state:
                    R[s, a] = self.rewards['middle']
                else:
                    R[s, a] = self.rewards['movement']
        
        return R

    def get_mdp_info(self):
        # Retourne les informations du MDP pour les algorithmes de Dynamic Programming
        return {
            'states': range(self.n_states),
            'actions': self.actions,
            'transition_matrix': self._transition_matrix,
            'reward_matrix': self._reward_matrix,
            'terminals': self.terminals,
            'gamma': 0.99
        }

    def visualisation(self):
        # Affiche l'état actuel de notre environnement
        print("\n" + "="*40)
        print("Grid World CORRIGÉ".center(40))
        print("-"*40)
        
        grid = [['[ ]' for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        
        # Marquer les terminaux
        for terminal in self.terminals:
            term_row, term_col = self._state_to_pos(terminal)
            grid[term_row][term_col] = '[T]'
        
        # Position actuelle
        agent_row, agent_col = self._state_to_pos(self.current_state)
        grid[agent_row][agent_col] = '[A]'
        
        # Afficher la grille avec bordures
        print('+' + '---+'*self.n_cols)
        for grid_row in grid:
            print('|' + '|'.join(grid_row) + '|')
            print('+' + '---+'*self.n_cols)
        
        print(f"Position de l'agent : ({agent_row},{agent_col})")
        print("="*40)

    def get_state_space(self):
        return range(self.n_states)

    def get_action_space(self):
        return self.actions

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property
    def reward_matrix(self):
        return self._reward_matrix


# Qu'est ce que le Monty Hall Paradox 1 dans notre cas ?

# L'agent est un candidat au jeu Monty Hall, Il doit prendre 2 décisions successives. 
# Dans cet environnement, il y a 3 portes A, B et C. 
# Au démarrage de l'environnement, une porte est tirée au hasard de manière cachée pour l'agent
# Il s'agit de la porte gagnante. 
# La première action de l'agent est de choisir une porte parmi les trois portes. 

# Ensuite, une porte parmi les 2 restantes non choisies par l'agent est retirée du jeu
# Il s'agit forcément d'une porte non gagnante. L'agent ensuite doit effectuer une nouvelle
# action : choisir de conserver la porte choisie au départ ou changer pour la porte restante. 
# Une fois le choix fait, la porte choisie est 'ouverte'
# On découvre si elle était gagnante (reward de 1.0) ou non (reward de 0.0).

# ENVIRONNEMENT MONTY HALL PARADOX 1

class MontyHallParadox1(BaseEnvironment):
    def __init__(self):
        # Configuration du jeu
        self.doors = [0, 1, 2]  # Portes A, B, C
        self.R = [0.0, 1.0]  # Récompenses possibles
        self.A = [0, 1]  # 0: rester, 1: changer
        
        # État INTERNE (caché à l'agent) - pour la simulation du jeu
        self._winning_door = None  # Information cachée
        self._revealed_door = None
        self._first_choice = None
        
        # État OBSERVABLE par l'agent (sans information de la porte gagnante)
        self.current_state = None  # (chosen, revealed) ou état initial
        
        # Métriques de performance
        self.total_games = 0
        self.wins = 0
        self.stay_wins = 0
        self.switch_wins = 0
        self.total_reward = 0
        
        # Construction des matrices pour DP avec nouveaux états
        self.states, self.state_to_idx = self._build_states()
        self._transition_matrix = self._build_transition_matrix()
        self._reward_matrix = self._build_reward_matrix()
        
        # Maintenant on peut faire un vrai reset
        self.reset()
        self._observation_space = type('obj', (), {'n': len(self.states)})()
        self._action_space = type('obj', (), {'n': 2})()  # Toujours 2 actions : 0 (rester), 1 (changer)

    def state_to_index(self, state):
        return self.state_to_idx[state]

    def index_to_state(self, idx):
        return self.states[idx]

    def reset(self, as_index=False):
        """Réinitialise l'environnement pour un nouvel épisode."""
        self._winning_door = random.choice(self.doors)
        self._revealed_door = None
        self._first_choice = None
        
        # État observable : juste un indicateur qu'on est au début
        self.current_state = "initial"  
        if as_index:
            return self.state_to_index(self.current_state)
        return self.current_state

    def step(self, action, is_dynamic_programming=False, as_index=False):
        """Exécute une action dans l'environnement."""
        if self.current_state == "terminal":
            if as_index:
                return self.state_to_index(self.current_state), 0.0, True, {}
            return self.current_state, 0.0, True, {}
        
        # Phase 1: Premier choix de porte
        if self.current_state == "initial":
            if action not in self.doors:
                raise ValueError(f"Action invalide {action}. Doit être entre 0 et 2.")
                
            # Révéler une porte non gagnante différente du choix
            available_doors = [d for d in self.doors if d != action and d != self._winning_door]
            self._revealed_door = random.choice(available_doors) if available_doors else None
            
            self._first_choice = action
            
            # État observable : seulement le choix et la porte révélée
            next_state = (action, self._revealed_door)
            
            if is_dynamic_programming:
                if as_index:
                    return self.state_to_index(next_state), 0.0, False
                return next_state, 0.0, False, {
                    'phase': 1,
                    'door_chosen': ['A', 'B', 'C'][action],
                    'door_revealed': ['A', 'B', 'C'][self._revealed_door] if self._revealed_door is not None else None
                }
            
            self.current_state = next_state
            if as_index:
                return self.state_to_index(next_state), 0.0, False, {
                    'phase': 1,
                    'door_chosen': ['A', 'B', 'C'][action],
                    'door_revealed': ['A', 'B', 'C'][self._revealed_door] if self._revealed_door is not None else None
                }
            return next_state, 0.0, False, {
                'phase': 1,
                'door_chosen': ['A', 'B', 'C'][action],
                'door_revealed': ['A', 'B', 'C'][self._revealed_door] if self._revealed_door is not None else None
            }
            
        # Phase 2: Décision de rester ou changer
        if action not in self.A:
            raise ValueError(f"Action invalide {action}. Doit être 0 (rester) ou 1 (changer).")
            
        chosen, revealed = self.current_state
        remaining = [d for d in self.doors if d != chosen and d != revealed][0]
        final_choice = chosen if action == 0 else remaining
        reward = 1.0 if final_choice == self._winning_door else 0.0
        
        if not is_dynamic_programming:
            # Mise à jour des métriques
            self.total_games += 1
            self.total_reward += reward
            if reward == 1.0:
                self.wins += 1
                if action == 0:
                    self.stay_wins += 1
                else:
                    self.switch_wins += 1
        
        next_state = "terminal"
        self.current_state = next_state
        
        if is_dynamic_programming:
            if as_index:
                return self.state_to_index(next_state), reward, True
            return next_state, reward, True, {
                'phase': 2,
                'final_choice': ['A', 'B', 'C'][final_choice],
                'winning_door': ['A', 'B', 'C'][self._winning_door],
                'action': 'resté' if action == 0 else 'changé'
            }
            
        if as_index:
            return self.state_to_index(next_state), reward, True, {
                'phase': 2,
                'final_choice': ['A', 'B', 'C'][final_choice],
                'winning_door': ['A', 'B', 'C'][self._winning_door],
                'action': 'resté' if action == 0 else 'changé'
            }
        return next_state, reward, True, {
            'phase': 2,
            'final_choice': ['A', 'B', 'C'][final_choice],
            'winning_door': ['A', 'B', 'C'][self._winning_door],
            'action': 'resté' if action == 0 else 'changé'
        }

    def _build_states(self):
        """Construit la liste des états OBSERVABLES et leur mapping vers des indices."""
        states = []
        state_to_idx = {}
        
        # État initial observable
        state = "initial"
        states.append(state)
        state_to_idx[state] = len(states) - 1
        
        # Phase 2 : États (chosen, revealed) - l'agent ne voit que ça
        for chosen in self.doors:
            for revealed in self.doors:
                if revealed != chosen:  # La porte révélée ne peut pas être celle choisie
                    state = (chosen, revealed)
                    states.append(state)
                    state_to_idx[state] = len(states) - 1
        
        # État terminal
        terminal_state = "terminal"
        states.append(terminal_state)
        state_to_idx[terminal_state] = len(states) - 1
        
        return states, state_to_idx

    def _build_transition_matrix(self):
        """Construit la matrice de transition P(s'|s,a) avec les nouveaux états."""
        n_states = len(self.states)
        n_actions = max(len(self.doors), len(self.A))
        P = np.zeros((n_states, n_actions, n_states))
        
        # Depuis l'état initial
        initial_idx = self.state_to_idx["initial"]
        for action in self.doors:
            # Pour chaque porte gagnante possible (équiprobable)
            for winning_door in self.doors:
                if action == winning_door:
                    # Si on choisit la porte gagnante, Monty peut révéler n'importe laquelle des autres
                    available_reveals = [d for d in self.doors if d != action]
                    for reveal in available_reveals:
                        next_state = (action, reveal)
                        next_idx = self.state_to_idx[next_state]
                        P[initial_idx, action, next_idx] += (1/3) * (1/len(available_reveals))
                else:
                    # Si on ne choisit pas la porte gagnante, Monty révèle l'autre porte non-gagnante
                    reveal = [d for d in self.doors if d != action and d != winning_door][0]
                    next_state = (action, reveal)
                    next_idx = self.state_to_idx[next_state]
                    P[initial_idx, action, next_idx] += (1/3)
        
        # Depuis les états de phase 2
        terminal_idx = self.state_to_idx["terminal"]
        for state in self.states:
            if isinstance(state, tuple) and len(state) == 2:  # États (chosen, revealed)
                state_idx = self.state_to_idx[state]
                for action in self.A:
                    P[state_idx, action, terminal_idx] = 1.0
        
        return P

    def _build_reward_matrix(self):
        """Construit la matrice de récompense R(s,a) avec les nouveaux états."""
        n_states = len(self.states)
        n_actions = max(len(self.doors), len(self.A))
        R = np.zeros((n_states, n_actions))
        
        # Les récompenses ne sont données qu'en phase 2
        for state in self.states:
            if isinstance(state, tuple) and len(state) == 2:  # États (chosen, revealed)
                state_idx = self.state_to_idx[state]
                chosen, revealed = state
                remaining = [d for d in self.doors if d != chosen and d != revealed][0]
                
                # Action 0 (rester) : récompense si chosen est gagnante
                R[state_idx, 0] = 1/3  # Espérance de gain en restant
                R[state_idx, 1] = 2/3  # Espérance de gain en changeant
        
        return R

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property  
    def reward_matrix(self):
        return self._reward_matrix

    def get_mdp_info(self):
        """Retourne les informations du MDP pour les algorithmes de DP."""
        return {
            'states': range(len(self._transition_matrix)),
            'actions': self.A,
            'transition_matrix': self._transition_matrix,
            'reward_matrix': self._reward_matrix,
            'terminals': [len(self._transition_matrix) - 1],  # Dernier état = terminal
            'gamma': 0.99
        }

    def get_state_space(self):
        return range(len(self.states))

    def get_action_space(self):
        return self.A

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


# Qu'est ce que le Monty Hall Paradox 2 dans notre cas ?

# Monty Hall "paradox" level 2 (description ci-dessous)  
# Il  s'agit  du  même  environnement  que  pré cé demment,  seulement  maintenant  5  portes  sont  
# disponibles,  et  l'agent  doit  effectuer  4  actions  successives  avant  l'ouverture  d'une  des  deux  portes  
# restantes.

# ENVIRONNEMENT MONTY HALL PARADOX 2

class MontyHallParadox2(BaseEnvironment):
    def __init__(self):
        # Configuration du jeu
        self.doors = [0, 1, 2, 3, 4]  # 5 Portes A, B, C, D, E
        self.R = [0.0, 1.0]  # Récompenses possibles
        self.A = [0, 1]  # 0: rester, 1: changer
        
        # État INTERNE (caché à l'agent) - pour la simulation du jeu
        self._winning_door = None  # Information cachée
        self._revealed_doors = []  # Portes révélées par Monty
        self._current_choice = None  # Porte actuellement choisie
        self._phase = 0  # Phase du jeu (0-3)
        
        # État OBSERVABLE par l'agent (sans information de la porte gagnante)
        self.current_state = None  # État observable par l'agent
        
        # Métriques de performance
        self.total_games = 0
        self.wins = 0
        self.stay_wins = 0
        self.switch_wins = 0
        self.total_reward = 0
        
        # Construction des matrices pour DP avec nouveaux états
        self.states, self.state_to_idx = self._build_states()
        self.transition_matrix = self._build_transition_matrix()
        self.reward_matrix = self._build_reward_matrix()
        
        # Maintenant on peut faire un vrai reset
        self.reset()
        self._observation_space = type('obj', (), {'n': len(self.states)})()
        self._action_space = type('obj', (), {'n': 2})()  # Toujours 2 actions : 0 (rester), 1 (changer)

    def state_to_index(self, state):
        return self.state_to_idx[state]

    def index_to_state(self, idx):
        return self.states[idx]

    def reset(self, as_index=False):
        """Réinitialise l'environnement pour un nouvel épisode."""
        self._winning_door = random.choice(self.doors)
        self._revealed_doors = []
        self._current_choice = None
        self._phase = 0
        self.current_state = "initial"
        if as_index:
            return self.state_to_index(self.current_state)
        return self.current_state

    def step(self, action, is_dynamic_programming=False, as_index=False):
        """Exécute une action dans l'environnement."""
        if self.current_state == "terminal":
            if as_index:
                return self.state_to_index(self.current_state), 0.0, True, {}
            return self.current_state, 0.0, True, {}
            
        # Phase finale (4ème action) - décision rester/changer
        if self._phase == 3:
            if action not in self.A:
                raise ValueError(f"Phase finale : l'action doit être 0 (rester) ou 1 (changer), pas {action}")
            
            available = [d for d in self.doors if d not in self._revealed_doors and d != self._current_choice]
            final_choice = self._current_choice if action == 0 else available[0]
            reward = 1.0 if final_choice == self._winning_door else 0.0
            
            if not is_dynamic_programming:
                self.total_games += 1
                self.total_reward += reward
                if reward == 1.0:
                    self.wins += 1
                    if action == 0:
                        self.stay_wins += 1
                    else:
                        self.switch_wins += 1
            
            next_state = "terminal"
            self.current_state = next_state
            
            if is_dynamic_programming:
                if as_index:
                    return self.state_to_index(next_state), reward, True, {
                        'phase': self._phase + 1,
                        'final_choice': ['A', 'B', 'C', 'D', 'E'][final_choice],
                        'winning_door': ['A', 'B', 'C', 'D', 'E'][self._winning_door],
                        'action': 'resté' if action == 0 else 'changé'
                    }
                return next_state, reward, True, {
                    'phase': self._phase + 1,
                    'final_choice': ['A', 'B', 'C', 'D', 'E'][final_choice],
                    'winning_door': ['A', 'B', 'C', 'D', 'E'][self._winning_door],
                    'action': 'resté' if action == 0 else 'changé'
                }
                
            if as_index:
                return self.state_to_index(next_state), reward, True, {
                    'phase': self._phase + 1,
                    'final_choice': ['A', 'B', 'C', 'D', 'E'][final_choice],
                    'winning_door': ['A', 'B', 'C', 'D', 'E'][self._winning_door],
                    'action': 'resté' if action == 0 else 'changé'
                }
            return next_state, reward, True, {
                'phase': self._phase + 1,
                'final_choice': ['A', 'B', 'C', 'D', 'E'][final_choice],
                'winning_door': ['A', 'B', 'C', 'D', 'E'][self._winning_door],
                'action': 'resté' if action == 0 else 'changé'
            }
        
        # Phases 1-3 : Choix de portes
        if action not in self.doors:
            raise ValueError(f"Phase {self._phase + 1} : l'action doit être une porte (0-4), pas {action}")
        
        if action in self._revealed_doors:
            raise ValueError(f"Phase {self._phase + 1} : la porte {['A', 'B', 'C', 'D', 'E'][action]} a déjà été révélée!")
            
        # Mise à jour du choix courant
        self._current_choice = action
        
        # Révélation d'une nouvelle porte par Monty
        available_for_reveal = [d for d in self.doors 
                               if d != action and d != self._winning_door and d not in self._revealed_doors]
        if available_for_reveal:
            revealed = random.choice(available_for_reveal)
            self._revealed_doors.append(revealed)
        
        self._phase += 1
        
        # État observable : (phase, current_choice, revealed_doors) - sans winning_door
        next_state = (self._phase, self._current_choice, tuple(sorted(self._revealed_doors)))
        
        if is_dynamic_programming:
            if as_index:
                return self.state_to_index(next_state), 0.0, False
            return next_state, 0.0, False, {
                'phase': self._phase,
                'door_chosen': ['A', 'B', 'C', 'D', 'E'][action],
                'door_revealed': ['A', 'B', 'C', 'D', 'E'][revealed] if available_for_reveal else None
            }
        
        self.current_state = next_state
        if as_index:
            return self.state_to_index(next_state), 0.0, False, {
                'phase': self._phase,
                'door_chosen': ['A', 'B', 'C', 'D', 'E'][action],
                'door_revealed': ['A', 'B', 'C', 'D', 'E'][revealed] if available_for_reveal else None
            }
        return next_state, 0.0, False, {
            'phase': self._phase,
            'door_chosen': ['A', 'B', 'C', 'D', 'E'][action],
            'door_revealed': ['A', 'B', 'C', 'D', 'E'][revealed] if available_for_reveal else None
        }

    def _build_states(self):
        """Construit la liste des états OBSERVABLES et leur mapping vers des indices."""
        states = []
        state_to_idx = {}
        
        # État initial observable
        state = "initial"
        states.append(state)
        state_to_idx[state] = len(states) - 1
        
        # États intermédiaires : (phase, current_choice, revealed_doors)
        # Phase 1-3, pour chaque combinaison possible
        for phase in range(1, 4):
            for current_choice in self.doors:
                # Générer toutes les combinaisons possibles de portes révélées pour cette phase
                for revealed_combo in combinations([d for d in self.doors if d != current_choice], phase):
                    state = (phase, current_choice, tuple(sorted(revealed_combo)))
                    states.append(state)
                    state_to_idx[state] = len(states) - 1
        
        # État terminal
        terminal_state = "terminal"
        states.append(terminal_state)
        state_to_idx[terminal_state] = len(states) - 1
        
        return states, state_to_idx

    def _build_transition_matrix(self):
        """Construit la matrice de transition P(s'|s,a) avec les nouveaux états."""
        n_states = len(self.states)
        n_actions = len(self.doors)  # Maximum entre choix de portes et rester/changer
        P = np.zeros((n_states, n_actions, n_states))
        
        # Depuis l'état initial - choix de la première porte
        initial_idx = self.state_to_idx["initial"]
        for action in self.doors:
            # Pour chaque porte gagnante possible (équiprobable)
            for winning_door in self.doors:
                # Monty révèle une porte non-gagnante différente du choix
                available_reveals = [d for d in self.doors if d != action and d != winning_door]
                if available_reveals:
                    for reveal in available_reveals:
                        next_state = (1, action, (reveal,))
                        if next_state in self.state_to_idx:
                            next_idx = self.state_to_idx[next_state]
                            P[initial_idx, action, next_idx] += (1/5) * (1/len(available_reveals))
        
        # Depuis les états intermédiaires
        for state in self.states:
            if isinstance(state, tuple) and len(state) == 3:  # États (phase, choice, revealed)
                phase, current_choice, revealed = state
                state_idx = self.state_to_idx[state]
                
                if phase < 3:  # Phases 1-2 : continuer à choisir des portes
                    for action in self.doors:
                        if action not in revealed and action != current_choice:
                            # Pour chaque porte gagnante possible
                            for winning_door in self.doors:
                                available_reveals = [d for d in self.doors 
                                                if d != action and d != winning_door and d not in revealed]
                                if available_reveals:
                                    for reveal in available_reveals:
                                        new_revealed = tuple(sorted(list(revealed) + [reveal]))
                                        next_state = (phase + 1, action, new_revealed)
                                        if next_state in self.state_to_idx:
                                            next_idx = self.state_to_idx[next_state]
                                            P[state_idx, action, next_idx] += (1/5) * (1/len(available_reveals))
                
                elif phase == 3:  # Phase finale : rester/changer
                    terminal_idx = self.state_to_idx["terminal"]
                    for action in self.A:
                        if action < n_actions:  # Vérifier que l'action est dans la plage
                            P[state_idx, action, terminal_idx] = 1.0
        
        return P

    def _build_reward_matrix(self):
        """Construit la matrice de récompense R(s,a) avec les nouveaux états."""
        n_states = len(self.states)
        n_actions = len(self.doors)
        R = np.zeros((n_states, n_actions))
        
        # Les récompenses ne sont données qu'en phase finale
        for state in self.states:
            if isinstance(state, tuple) and len(state) == 3:  # États (phase, choice, revealed)
                phase, current_choice, revealed = state
                if phase == 3:  # Phase finale
                    state_idx = self.state_to_idx[state]
                    available = [d for d in self.doors if d not in revealed and d != current_choice]
                    
                    if available:
                        # Action 0 (rester) : probabilité 1/5 que current_choice soit gagnante
                        R[state_idx, 0] = 1/5
                        
                        # Action 1 (changer) : probabilité 4/5 que la porte restante soit gagnante
                        R[state_idx, 1] = 4/5
        
        return R

    def get_mdp_info(self):
        """Retourne les informations MDP pour les algorithmes de DP."""
        return {
            'states': list(range(len(self.states))),  # Indices numériques [0, 1, 2, ...]
            'actions': list(range(len(self.doors))),  # [0, 1, 2, 3, 4]
            'transition_matrix': self.transition_matrix,
            'reward_matrix': self.reward_matrix,
            'terminals': [self.state_to_idx["terminal"]],
            'gamma': 0.99
        }

    def get_state_space(self):
        return range(len(self.states))

    def get_action_space(self):
        return self.A

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


# Two Round Rock Paper Scisssors
# L'agent fait une partie constituée de 2 rounds de Pierre Feuille Ciseaux 
# Face à un adversaire qui joue de manière particulière
# Ce dernier joue aléatoirement au premier round
# Mais au deuxième round, il joue FORCEMENT le choix de l'agent au premier round

class RockPaperScissors(BaseEnvironment):
    # États du jeu
    INITIAL_STATE = 0
    ROUND1_START = 1
    ROUND2_START = 4
    TERMINAL_DP = 7
    TERMINAL_MC = 8

    def __init__(self):
        self.actions = [0, 1, 2]
        self.n_actions = len(self.actions)
        self.n_states = 9

        # Matrice du jeu de base
        self.reward_matrix = np.array([
            [0, -1, 1],   # Pierre
            [1, 0, -1],   # Feuille
            [-1, 1, 0]    # Ciseaux
        ])

        self.agent_action_round1 = None
        self.transition_matrix = self._build_transition_matrix()
        self.reward_matrix_dp = self._build_reward_matrix()

        self.steps_count = 0
        self.total_reward = 0
        self.wins = 0
        self.total_games = 0
        self.reset()
        self._observation_space = type('obj', (), {'n': self.n_states})()
        self._action_space = type('obj', (), {'n': self.n_actions})()

    def reset(self):
        self.current_round = 0
        self.previous_agent_action = None
        self.previous_opponent_action = None
        self.agent_action_round1 = None
        self.done = False
        self.steps_count = 0
        self.total_reward = 0
        return self._get_state()

    def step(self, action, is_dynamic_programming=False):
        if self.current_round == 0:
            opponent_action = random.choice(self.actions)
            self.agent_action_round1 = action
        else:
            opponent_action = self.agent_action_round1
            
        reward = self.reward_matrix[action, opponent_action]
        
        self.previous_agent_action = action
        self.previous_opponent_action = opponent_action
        self.current_round += 1
        self.done = self.current_round >= 2
        
        self.steps_count += 1
        self.total_reward += reward
        if reward > 0:
            self.wins += 1
        if self.done:
            self.total_games += 1

        if is_dynamic_programming:
            return self._get_state(for_dp=True), reward, self.done
        else:
            return self._get_state(for_dp=False), reward, self.done, {
                'round': self.current_round,
                'agent_action': action,
                'opponent_action': opponent_action
            }

    def _get_state(self, for_dp=False):
        if for_dp:
            if self.current_round == 0:
                return self.INITIAL_STATE
            elif self.current_round == 1:
                return self.ROUND1_START + self.previous_opponent_action
            elif self.current_round == 2 and not self.done:
                return self.ROUND2_START + self.agent_action_round1
            else:
                return self.TERMINAL_DP
        else:
            if self.current_round == 0:
                return self.INITIAL_STATE
            elif self.current_round == 1:
                return self.ROUND1_START + self.previous_opponent_action
            elif self.current_round == 2 and not self.done:
                return self.ROUND2_START + self.agent_action_round1
            else:
                return self.TERMINAL_MC

    def _build_transition_matrix(self):
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        # Depuis l'état initial (0)
        for action in self.actions:
            for opp_action in self.actions:
                next_state = self.ROUND1_START + opp_action
                P[self.INITIAL_STATE, action, next_state] = 1/3
        
        # Depuis les états après premier round (1-3)
        for state in range(self.ROUND1_START, self.ROUND2_START):
            for action in self.actions:
                next_state = self.ROUND2_START + action
                P[state, action, next_state] = 1.0
        
        # Depuis les états du second round (4-6)
        for state in range(self.ROUND2_START, self.TERMINAL_DP):
            for action in self.actions:
                P[state, action, self.TERMINAL_DP] = 1.0
        
        return P

    def _build_reward_matrix(self):
        R = np.zeros((self.n_states, self.n_actions))
        
        # APPROCHE 1: Récompenses asymétriques à l'état initial
        # Simuler une légère préférence sans donner la solution optimale
        R[self.INITIAL_STATE, 0] = 0.5   # Pierre légèrement avantagé
        R[self.INITIAL_STATE, 1] = 0.3   # Feuille intermédiaire  
        R[self.INITIAL_STATE, 2] = 0.1   # Ciseaux moins avantagé
        
        # États 1-3: Récompenses du round 1 seulement
        for state in range(self.ROUND1_START, self.ROUND2_START):
            opponent_action = state - self.ROUND1_START
            for action in self.actions:
                R[state, action] = self.reward_matrix[action, opponent_action]
        
        # États 4-6: Récompenses du round 2
        for state in range(self.ROUND2_START, self.TERMINAL_DP):
            agent_previous_action = state - self.ROUND2_START
            for action in self.actions:
                R[state, action] = self.reward_matrix[action, agent_previous_action]
        
        return R

    def get_mdp_info(self):
        return {
            'states': range(self.n_states),
            'actions': self.actions,
            'transition_matrix': self.transition_matrix,
            'reward_matrix': self.reward_matrix_dp,
            'terminals': [self.TERMINAL_DP],
            'gamma': 0.9
        }

    def get_state_space(self):
        return range(self.n_states)

    def get_action_space(self):
        return self.actions

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


# Création d'un fichier __init__.py pour faire du dossier game un module Python
def __init__():
    pass 
