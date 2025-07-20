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

class LineWorld(BaseEnvironment):
    def __init__(self, length=8):
        self.length = length
        self.start = length // 2
        self.terminals = [0, length-1]
        self.actions = [0, 1]
        self.reset()

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
        # une visualisation simple de l'environnement
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
        # Cette méthode est utilisée uniquement par les algorithmes de programmation dynamique.
        P = np.zeros((self.length, len(self.actions), self.length))
        
        for s in range(self.length):
            # Action gauche (0)
            # on ne peut pas aller à gauche si on est à l'état 0
            # P[s, 0, next_s] retourne la probabilité de transition de l'état s à l'état next_s
            # en effectuant l'action gauche
            next_s = max(0, s - 1)
            P[s, 0, next_s] = 1.0
            
            # Action droite (1)
            # on ne peut pas aller à droite si on est à l'état terminal
            # P[s, 1, next_s] retourne la probabilité de transition de l'état s à l'état next_s
            # en effectuant l'action droite
            next_s = min(self.length-1, s+1)
            P[s, 1, next_s] = 1.0
            
        return P

    def _build_reward_matrix(self):
        # Construit la matrice de récompense R(s,a).
        # Cette méthode est utilisée uniquement par les algorithmes de programmation dynamique.
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


    # Méthodes pour les algorithmes de programmation dynamique
    def get_mdp_info(self):
        # Retourne les informations du MDP
        # Cette méthode est utilisée uniquement par les algorithmes de programmation dynamique.
        return {
            'states': range(self.length),
            'actions': self.actions,
            'transition_matrix': self._transition_matrix,
            'reward_matrix': self._reward_matrix,
            'terminals': self.terminals,
            'gamma': 0.99  
            # nous optons pour un facteur d'actualisation de 0.99
            # parce que c'est un facteur d'actualisation assez élevé
            # la nature de notre environnement est telle que le risque de
            # divergence est très très faible.
        }
    

    def jeu_manuel(self):
        # Cette méthode permet à un utilisateur de jouer contre l'environnement
        self.reset()
        print("\n=== Line World ===")
        print("Actions: 0 (gauche) ou 1 (droite)")
        print("Votre objectif à vous est d'atteindre une extrémité de la ligne")
        
        while True:
            self.visualisation()
            print(f"\nPosition actuelle: {self.state}")
            
            try:
                action = int(input("Action (0/1): "))
                if action not in self.actions:
                    print("Action invalide! Utilisez 0 ou 1")
                    continue
            except ValueError:
                print("Entrée invalide! Entrez un nombre")
                continue
                
            next_state, reward, done = self.step(action)
            print(f"Récompense: {reward}")
            
            if done:
                self.visualisation()
                print("Terminal atteint!")
                break

    
    def get_metrics(self):
        # Retourne les métriques de performance
        return {
            'current_state': self.state,
            'episode': self.episode_count,
            'steps': self.steps_count,
            'total_reward': self.total_reward,
            'average_reward': self.total_reward / max(1, self.steps_count),
            'visited_states': len(self.visited_states),
            'state_coverage': len(self.visited_states) / self.length,
            'distance_to_goal': min(abs(self.state - t) for t in self.terminals),
            'episode_history': self.episode_history
        }


    def replay(self, strategy, delay=1):
        """Rejoue une séquence d'actions pas à pas.
        
        Args:
            strategy (list): Liste d'actions (0: gauche, 1: droite)
            delay (int): Délai en secondes entre chaque action (par défaut 1s)
        """
        import time
        
        if not strategy:
            raise ValueError("La stratégie ne peut pas être vide")
            
        # Réinitialisation
        state = self.reset()
        total_reward = 0
        print("\n🎯 Position initiale")
        self.visualisation()
        time.sleep(delay)
        
        # Exécution de chaque action
        for i, action in enumerate(strategy, 1):
            print(f"\n🎯 Action {i}/{len(strategy)}")
            state, reward, done, info = self.step(action)
            total_reward += reward
            
            print(f"\nAction : {'Gauche' if action == 0 else 'Droite'}")
            print(f"Position : {info['position']}")
            print(f"Récompense : {reward:.1f}")
            self.visualisation()
            time.sleep(delay)
            
            if done:
                print(f"\n🏁 Terminal atteint ! Récompense totale : {total_reward:.1f}")
                break
                
        return total_reward

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
        
        # Définissons les récompenses
        self.rewards = {
            'movement': -1.0,  # Déplacement normal
            'middle': -3.0,    # Vers le milieu (pénalité plus forte)
            'no_move': -2.0    # Pas de mouvement possible
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
        
        # Calcul de la récompense
        if (new_row, new_col) == (current_row, current_col):
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
        # Construit la matrice des récompenses R(s,a)
        R = np.zeros((self.n_states, len(self.actions)))
        
        for s in range(self.n_states):
            if s in self.terminals:
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
                
                if (new_row, new_col) == (row, col):
                    R[s, a] = self.rewards['no_move']
                elif next_state == self.start_state:
                    R[s, a] = self.rewards['middle']
                else:
                    R[s, a] = self.rewards['movement']
        
        return R


    def get_mdp_info(self):
        # Retourne les informations du MDP pour les algorithmes de Dynamic Programming
        # Value iteration et policy iteration
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
        print("Grid World".center(40))
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


    def jeu_manuel(self):
        # Cette méthode permet à un utilisateur de jouer contre l'environnement
        # Elle réponds à l'exigence: "Il devra aussi être possible de pouvoir agir 'manuellement' (agent humain) sur chaque environnement
        # pour pouvoir s'assurer du bon respect des règles de ces derniers."
        self.reset()
        total_reward = 0
        
        print("\n=== Grid World ===")
        print("Actions disponibles :")
        print("0: Gauche  ⬅️")
        print("1: Droite  ➡️")
        print("2: Haut    ⬆️")
        print("3: Bas     ⬇️")
        print("\nObjectif: Atteignez une sortie(T), pas si dur non ?")
        
        while True:
            self.visualisation()
            
            try:
                action = int(input("\nAction (0-3): "))
                if action not in self.actions:
                    print("Action invalide! Choisissez entre 0 et 3")
                    continue
            except ValueError:
                print("Entrée invalide! Entrez un nombre")
                continue
            
            next_state, reward, done, info = self.step(action)
            total_reward += reward
            
            print(f"Récompense: {reward}")
            
            if done:
                self.visualisation()
                print(f"\nTerminal atteint! Récompense totale: {total_reward}")
                break

    
    def get_metrics(self):
        # Retourne des métriques pour connaitre les performmances de l'agent
        current_row, current_col = self._state_to_pos(self.current_state)
        return {
            'current_state': self.current_state,
            'steps_taken': self.steps_count,
            'total_reward': self.total_reward,
            'visited_states': self.visited_states,
            'average_reward': self.total_reward / max(1, self.steps_count)
        }


    def reset_metrics(self):
        # Reinitialise les métriques pour une nouvelle expérience
        self.steps_count = 0
        self.total_reward = 0
        self.visited_states = set()

    
    def replay(self, strategy, delay=1):
        """Rejoue une séquence d'actions pas à pas.
        
        Args:
            strategy (list): Liste d'actions (0: gauche, 1: droite, 2: haut, 3: bas)
            delay (int): Délai en secondes entre chaque action (par défaut 1s)
        """
        import time
        
        if not strategy:
            raise ValueError("La stratégie ne peut pas être vide")
            
        # Réinitialisation
        state = self.reset()
        total_reward = 0
        print("\n🎯 Position initiale")
        self.visualisation()
        time.sleep(delay)
        
        # Exécution de chaque action
        action_names = ['Gauche', 'Droite', 'Haut', 'Bas']
        for i, action in enumerate(strategy, 1):
            print(f"\n🎯 Action {i}/{len(strategy)}")
            state, reward, done, info = self.step(action)
            total_reward += reward
            
            print(f"\nAction : {action_names[action]}")
            print(f"Position : {info['position']}")
            print(f"Récompense : {reward:.1f}")
            self.visualisation()
            time.sleep(delay)
            
            if done:
                print(f"\n🏁 Terminal atteint ! Récompense totale : {total_reward:.1f}")
                break
                
        return total_reward

    
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


    def reset(self):
        """Réinitialise l'environnement pour un nouvel épisode."""
        self._winning_door = random.choice(self.doors)
        self._revealed_door = None
        self._first_choice = None
        
        # État observable : juste un indicateur qu'on est au début
        self.current_state = "initial"  
        return self.current_state


    def step(self, action, is_dynamic_programming=False):
        """Exécute une action dans l'environnement."""
        if self.current_state == "terminal":
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
                return next_state, 0.0, False
            
            self.current_state = next_state
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
            return next_state, reward, True
            
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
                
                # Pour calculer l'espérance, on considère toutes les portes gagnantes possibles
                # Probabilité conditionnelle sachant (chosen, revealed)
                
                # Action 0 (rester) : récompense si chosen est gagnante
                # P(winning=chosen | chosen, revealed) = ?
                # Si revealed != chosen, alors soit winning=chosen, soit winning=remaining
                # Par équiprobabilité initiale et révélation de Monty :
                # - Si chosen était gagnante (prob 1/3), Monty révèle revealed
                # - Si remaining était gagnante (prob 1/3), Monty révèle revealed  
                # - Si revealed était gagnante (prob 1/3), impossible car Monty ne révèle jamais la gagnante
                
                # Donc P(winning=chosen | chosen, revealed) = 1/3 / (1/3 + 1/3) = 1/2 ? NON !
                # En fait, le calcul correct :
                # P(winning=chosen | chosen, revealed) = 1/3 (probabilité a priori)
                # P(winning=remaining | chosen, revealed) = 2/3 (par complémentaire)
                
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


    def visualisation(self):
        """Affiche l'état actuel du jeu."""
        print("\n" + "="*50)
        print("🚪 MONTY HALL 🚪".center(50))
        print("-"*50)
        
        if self.current_state == "terminal":
            print("Partie terminée !".center(50))
            print(f"La porte gagnante était : {['A', 'B', 'C'][self._winning_door]}")
            return
            
        doors = ['[A]', '[B]', '[C]']
        
        if self.current_state == "initial":
            print("Choisissez une porte :".center(50))
            print(" ".join(doors).center(50))
        else:
            chosen, revealed = self.current_state
            doors[revealed] = '[X]'  # Porte révélée
            doors[chosen] = '[*]'  # Porte choisie
            print("X : Porte révélée (vide)".center(50))
            print("* : Votre choix initial".center(50))
            print(" ".join(doors).center(50))
            print("\nVoulez-vous :")
            print("0 : Rester sur votre choix")
            print("1 : Changer de porte")
            
        print("="*50)


    def jeu_manuel(self):
        """Permet à un utilisateur de jouer une partie."""
        self.reset()
        print("\n=== Monty Hall ===")
        print("Bienvenue dans le paradoxe de Monty Hall!")
        print("Il y a trois portes : A(0), B(1) et C(2)")
        print("Derrière l'une d'elles se trouve le grand prix!")
        
        while True:
            self.visualisation()
            
            if self.current_state == "initial":
                try:
                    action = int(input("\nChoisissez une porte (0-2): "))
                    if action not in self.doors:
                        print("Action invalide! Choisissez entre 0 et 2")
                        continue
                except ValueError:
                    print("Entrée invalide! Entrez un nombre")
                    continue
            else:
                try:
                    action = int(input("\nVotre décision (0: rester, 1: changer): "))
                    if action not in self.A:
                        print("Action invalide! Choisissez 0 ou 1")
                        continue
                except ValueError:
                    print("Entrée invalide! Entrez un nombre")
                    continue
            
            _, reward, done, info = self.step(action)
            
            if done:
                self.visualisation()
                print(f"\nRésultat: {'Gagné!' if reward == 1.0 else 'Perdu!'}")
                print(f"La porte gagnante était : {info['winning_door']}")
                print(f"Vous avez {info['action']} et choisi la porte {info['final_choice']}")
                break


    def get_metrics(self):
        """Retourne les métriques de performance."""
        return {
            'total_games': self.total_games,
            'wins': self.wins,
            'win_rate': self.wins / max(1, self.total_games),
            'stay_wins': self.stay_wins,
            'switch_wins': self.switch_wins,
            'stay_win_rate': self.stay_wins / max(1, self.total_games),
            'switch_win_rate': self.switch_wins / max(1, self.total_games),
            'total_reward': self.total_reward,
            'average_reward': self.total_reward / max(1, self.total_games)
        }


    def replay(self, strategy, delay=2):
        """Rejoue une séquence d'actions pas à pas.
        
        Args:
            strategy (list): Liste de 2 actions [première_porte, rester_ou_changer]
                           première_porte: 0, 1, ou 2 (A, B, ou C)
                           rester_ou_changer: 0 (rester) ou 1 (changer)
            delay (int): Délai en secondes entre chaque action (par défaut 2s)
        """
        import time
        
        if len(strategy) != 2:
            raise ValueError("La stratégie doit contenir exactement 2 actions pour Monty Hall")
            
        # Réinitialisation
        state = self.reset()
        print("\n🎯 Début de la partie")
        self.visualisation()
        time.sleep(delay)
        
        # Première action : choix de la porte
        print("\n🎯 Phase 1 : Premier choix")
        state, reward, done, info = self.step(strategy[0])
        print(f"\nPorte choisie : {info['door_chosen']}")
        print(f"Monty révèle : {info['door_revealed']} (contient une chèvre)")
        self.visualisation()
        time.sleep(delay)
        
        # Deuxième action : rester ou changer
        print("\n🎯 Phase 2 : Décision finale")
        state, reward, done, info = self.step(strategy[1])
        self.visualisation()
        print(f"\nRésultat final:")
        print(f"- Porte gagnante : {info['winning_door']}")
        print(f"- Votre choix final : {info['final_choice']}")
        print(f"- Vous avez {info['action']}")
        print(f"- Récompense : {reward:.1f}")
        
        return reward

    def get_state_space(self):
        return self.states

    def get_action_space(self):
        return self.doors + self.A


# Qu'est ce que le Monty Hall Paradox 2 dans notre cas ?

# Monty Hall "paradox" level 2 (description ci-dessous)  
# Il  s'agit  du  même  environnement  que  pré cé demment,  seulement  maintenant  5  portes  sont  
# disponibles,  et  l'agent  doit  effectuer  4  actions  successives  avant  l'ouverture  d'une  des  deux  portes  
# restantes.

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


    def reset(self):
        """Réinitialise l'environnement pour un nouvel épisode."""
        self._winning_door = random.choice(self.doors)
        self._revealed_doors = []
        self._current_choice = None
        self._phase = 0
        
        # État observable : juste un indicateur qu'on est au début
        self.current_state = "initial"
        return self.current_state


    def step(self, action, is_dynamic_programming=False):
        """Exécute une action dans l'environnement."""
        if self.current_state == "terminal":
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
                return next_state, reward, True
                
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
            return next_state, 0.0, False
        
        self.current_state = next_state
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
                from itertools import combinations
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
                        # Mais il n'y a qu'une seule porte restante, donc elle a toute la probabilité
                        # des portes non-choisies initialement
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


    def get_action_space_size(self):
        """Retourne la taille de l'espace d'action."""
        return len(self.doors)  # 5 actions possibles


    def get_state_space_size(self):
        """Retourne la taille de l'espace d'état."""
        return len(self.states)


    def state_to_index(self, state):
        """Convertit un état en index."""
        return self.state_to_idx.get(state, -1)


    def index_to_state(self, index):
        """Convertit un index en état."""
        if 0 <= index < len(self.states):
            return self.states[index]
        return None


    def visualisation(self):
        """Affiche l'état actuel du jeu."""
        print("\n" + "="*50)
        print("🎮 MONTY HALL 2 - 5 PORTES 🎮".center(50))
        print("-"*50)
        
        if self.current_state == "terminal":
            print("Partie terminée !".center(50))
            print(f"La porte gagnante était : {['A', 'B', 'C', 'D', 'E'][self._winning_door]}")
            return
            
        doors = ['[A]', '[B]', '[C]', '[D]', '[E]']
        revealed = self._revealed_doors
        
        # Marquer les portes révélées et le choix actuel
        if revealed:
            for r in revealed:
                doors[r] = '[X]'  # Porte révélée
        if self._current_choice is not None:
            doors[self._current_choice] = '[*]'  # Porte choisie
        
        print(f"Phase {self._phase + 1}/4".center(50))
        if self._phase < 3:
            print("Choisissez une porte :".center(50))
        else:
            print("Décision finale :".center(50))
            print("0 : Rester sur votre choix".center(50))
            print("1 : Changer de porte".center(50))
        
        print("X : Porte révélée (vide)".center(50))
        print("* : Votre choix actuel".center(50))
        print(" ".join(doors).center(50))
        print("="*50)


    def jeu_manuel(self):
        """Permet à un utilisateur de jouer une partie."""
        self.reset()
        print("\n=== Monty Hall 2 - 5 Portes ===")
        print("Bienvenue dans le paradoxe de Monty Hall version 2!")
        print("Il y a cinq portes : A(0), B(1), C(2), D(3), E(4)")
        print("Derrière l'une d'elles se trouve le grand prix!")
        print("\nVous devez faire 4 choix successifs :")
        print("- Les 3 premiers sont des choix de porte")
        print("- Le dernier est la décision de rester ou changer")
        
        while True:
            self.visualisation()
            
            try:
                if self._phase < 3:
                    action = int(input(f"\nChoisissez une porte (0-4): "))
                    if action not in self.doors:
                        print("Action invalide! Choisissez entre 0 et 4")
                        continue
                    if action in self._revealed_doors:
                        print("Cette porte a déjà été révélée!")
                        continue
                else:
                    action = int(input("\nVotre décision finale (0: rester, 1: changer): "))
                    if action not in self.A:
                        print("Action invalide! Choisissez 0 ou 1")
                        continue
            except ValueError:
                print("Entrée invalide! Entrez un nombre")
                continue
            
            _, reward, done, info = self.step(action)
            
            if done:
                self.visualisation()
                print(f"\nRésultat: {'Gagné!' if reward == 1.0 else 'Perdu!'}")
                print(f"La porte gagnante était : {info['winning_door']}")
                print(f"Vous avez {info['action']} et choisi la porte {info['final_choice']}")
                break


    def replay(self, strategy, delay=2):
        """Rejoue une séquence d'actions pas à pas.
        
        Args:
            strategy (list): Liste de 4 actions [porte1, porte2, porte3, rester_ou_changer]
                           porte1, porte2, porte3: 0-4 (A-E)
                           rester_ou_changer: 0 (rester) ou 1 (changer)
            delay (int): Délai en secondes entre chaque action (par défaut 2s)
        """
        import time
        
        if len(strategy) != 4:
            raise ValueError("La stratégie doit contenir exactement 4 actions pour Monty Hall 2")
            
        # Réinitialisation
        state = self.reset()
        print("\n🎯 Début de la partie")
        self.visualisation()
        time.sleep(delay)
        
        # Trois premiers choix
        for i in range(3):
            print(f"\n🎯 Phase {i+1} : Choix de porte")
            state, reward, done, info = self.step(strategy[i])
            print(f"\nPorte choisie : {info['door_chosen']}")
            print(f"Monty révèle : {info['door_revealed']} (contient une chèvre)")
            self.visualisation()
            time.sleep(delay)
        
        # Décision finale
        print("\n🎯 Phase 4 : Décision finale")
        state, reward, done, info = self.step(strategy[3])
        self.visualisation()
        print(f"\nRésultat final:")
        print(f"- Porte gagnante : {info['winning_door']}")
        print(f"- Votre choix final : {info['final_choice']}")
        print(f"- Vous avez {info['action']}")
        print(f"- Récompense : {reward:.1f}")
        
        return reward


# Two Round Rock Paper Scisssors
# L'agent fait une partie constituée de 2 rounds de Pierre Feuille Ciseaux 
# Face à un adversaire qui joue de manière particulière
# Ce dernier joue aléatoirement au premier round
# Mais au deuxième round, il joue FORCEMENT le choix de l'agent au premier round

class RockPaperScissors:
    # États du jeu
    INITIAL_STATE = 0
    ROUND1_START = 1  # États 1-3 basés sur l'action adversaire
    ROUND2_START = 4  # États 4-6 basés sur l'action agent round 1
    TERMINAL_DP = 7   # État terminal pour DP
    TERMINAL_MC = 8   # État terminal pour MC/TD

    def __init__(self):
        # Notre agent ne peut choissir que trois actions:
        # 0: Pierre
        # 1: Feuille
        # 2: Ciseaux
        self.actions = [0, 1, 2]
        self.n_actions = len(self.actions)
        
        # États:
        # 0: État initial
        # 1-3: Premier round après action adversaire (state-1 = action adversaire)
        # 4-6: Deuxième round (state-4 = action agent round 1)
        # 7: État terminal pour DP
        # 8: État terminal pour MC/TD
        self.n_states = 9

        # Matrice des récompenses du jeu
        self.reward_matrix = np.array([
            # Pierre  Feuille  Ciseaux
            [0,      -1,      1],     # Pierre
            [1,      0,       -1],    # Feuille
            [-1,     1,       0]      # Ciseaux
        ])

        # Pour suivre l'action de l'agent au premier round
        self.agent_action_round1 = None
        
        # Uniquement pour nos algorithmes de programmation dynamique
        self.transition_matrix = self._build_transition_matrix()
        self.reward_matrix_dp = self._build_reward_matrix()

        # Métriques
        self.steps_count = 0
        self.total_reward = 0
        self.wins = 0
        self.total_games = 0

        # Réinitialisation de l'environnement
        self.reset()

    def reset(self):
        self.current_round = 0
        self.previous_agent_action = None
        self.previous_opponent_action = None
        self.agent_action_round1 = None  # Important pour le suivi
        self.done = False
        
        # Réinitialisation des métriques d'épisode
        self.steps_count = 0
        self.total_reward = 0
        
        return self._get_state()

    def step(self, action, is_dynamic_programming=False):
        # Si le round est 0, nous sommes au début de la partie
        if self.current_round == 0:
            opponent_action = random.choice(self.actions)
            self.agent_action_round1 = action  # Stockage de l'action du round 1
        else:
            # Au round 2, l'adversaire joue l'action de l'agent du round 1
            opponent_action = self.agent_action_round1
            
        reward = self.reward_matrix[action, opponent_action]
        
        self.previous_agent_action = action
        self.previous_opponent_action = opponent_action
        self.current_round += 1
        self.done = self.current_round >= 2
        
        # Mise à jour des métriques
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
                'opponent_action': opponent_action,
                'action_name': ['Pierre', 'Feuille', 'Ciseaux'][action]
            }

    def _get_state(self, for_dp=False):
        if for_dp:
            if self.current_round == 0:
                return self.INITIAL_STATE
            elif self.current_round == 1:
                # État basé sur l'action de l'adversaire
                return self.ROUND1_START + self.previous_opponent_action
            elif self.current_round == 2 and not self.done:
                # État basé sur l'action de l'agent au round 1
                return self.ROUND2_START + self.agent_action_round1
            else:
                return self.TERMINAL_DP
        else:
            # Version numérique pour MC/TD Learning
            if self.current_round == 0:
                return self.INITIAL_STATE
            elif self.current_round == 1:
                return self.ROUND1_START + self.previous_opponent_action
            elif self.current_round == 2 and not self.done:
                return self.ROUND2_START + self.agent_action_round1
            else:
                return self.TERMINAL_MC

    def _build_transition_matrix(self):
        # Construit la matrice de transition P(s'|s,a)
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        # Depuis l'état initial (0)
        for action in self.actions:
            # L'adversaire joue aléatoirement
            for opp_action in self.actions:
                next_state = self.ROUND1_START + opp_action
                P[self.INITIAL_STATE, action, next_state] = 1/3
        
        # Depuis les états après premier round (1-3)
        for state in range(self.ROUND1_START, self.ROUND2_START):
            for action in self.actions:
                # L'action actuelle détermine l'état suivant
                next_state = self.ROUND2_START + action  # Action actuelle = action round 1
                P[state, action, next_state] = 1.0
        
        # Depuis les états du second round (4-6)
        for state in range(self.ROUND2_START, self.TERMINAL_DP):
            for action in self.actions:
                # Transition vers l'état terminal
                P[state, action, self.TERMINAL_DP] = 1.0
        
        return P


    def _build_reward_matrix(self):
        # Construit la matrice des récompenses pour la programmation dynamique
        R = np.zeros((self.n_states, self.n_actions))
        
        # État initial: récompense moyenne car l'adversaire joue aléatoirement
        for action in self.actions:
            R[self.INITIAL_STATE, action] = np.mean([self.reward_matrix[action, opp] 
                                for opp in self.actions])
        
        # États 1-3: récompense exacte basée sur l'action adversaire du round 1
        for state in range(self.ROUND1_START, self.ROUND2_START):
            opponent_action = state - self.ROUND1_START  # L'action de l'adversaire au round 1
            for action in self.actions:
                R[state, action] = self.reward_matrix[action, opponent_action]
        
        # États 4-6: récompense exacte car l'adversaire va jouer
        # l'action de l'agent du round 1 (déterministe)
        for state in range(self.ROUND2_START, self.TERMINAL_DP):
            agent_previous_action = state - self.ROUND2_START  # L'action de l'agent au round 1
            for action in self.actions:
                R[state, action] = self.reward_matrix[action, agent_previous_action]
        
        # États terminaux (7-8): récompenses = 0
        
        return R
    

    def jeu_manuel(self):
        # Cette méthode permet à un utilisateur de jouer contre l'environnement
        state = self.reset()
        total_reward = 0
        
        while True:
            self.visualisation()
            
            # Input de l'utilisateur
            print("\nChoisissez votre action:")
            print("0: Pierre 🪨")
            print("1: Feuille 📄")
            print("2: Ciseaux ✂️")
            
            try:
                action = int(input("Votre choix (0-2): "))
                if action not in self.actions:
                    print("Action invalide! Choisissez 0, 1 ou 2")
                    continue
            except ValueError:
                print("Entrée invalide! Entrez un nombre")
                continue
            
            # Exécuter l'action
            state, reward, done = self.step(action)
            total_reward += reward
            
            if done:
                self.visualisation()
                print(f"\nPartie terminée! Score total: {total_reward}")
                break


    def visualisation(self):
        # Séparateur pour plus de clarté
        print("\n" + "="*40)
        
        # Affichage correct des rounds (limité à 2)
        round_display = min(self.current_round + 1, 2)
        print(f"🎮 ROUND 🎮".center(40))
        print("-"*40)
        
        # Au début du jeu
        if self.previous_agent_action is None:
            print("🎲 Nouvelle partie ! 🎲".center(40))
            print("="*40 + "\n")
            return
        
        actions_emoji = {
            0: "🪨  Pierre",
            1: "📄  Feuille",
            2: "✂️  Ciseaux",
            None: "❓  Aucune"
        }
        
        print(f"Vous      : {actions_emoji[self.previous_agent_action]}")
        print(f"Adversaire: {actions_emoji[self.previous_opponent_action]}")
        print("-"*40)
        
        # Calcul et affichage du résultat
        reward = self.reward_matrix[self.previous_agent_action, 
                                self.previous_opponent_action]
        
        if reward > 0:
            result = "Winner winner chicken dinner ! 🎉"
        elif reward < 0:
            result = "Malheuresement nous ne pouvons faire suite à votre candidature, ah pardon vous avez juste perdu ! 💔"
        else:
            result = "It's a tie ! Wow 🤝"
        
        print(result.center(40))
        print("="*40 + "\n")


    def _state_to_index(self, round_num, prev_agent_action, prev_opponent_action):
        """Convertit un état en index unique pour les algorithmes tabulaires."""
        if prev_agent_action is None:
            prev_agent_action = 0
        if prev_opponent_action is None:
            prev_opponent_action = 0
        return round_num * 9 + prev_agent_action * 3 + prev_opponent_action


    def _index_to_state(self, index):
        """Convertit un index en état pour les algorithmes tabulaires."""
        round_num = index // 9
        remainder = index % 9
        prev_agent_action = remainder // 3
        prev_opponent_action = remainder % 3
        return round_num, prev_agent_action, prev_opponent_action


    def get_metrics(self):
        # Retourne des métriques pour connaitre les performmances de l'agent
        return {
            'current_round': self.current_round,
            'previous_actions': {
                'agent': self.previous_agent_action,
                'opponent': self.previous_opponent_action
            },
            'total_reward': self.total_reward,
            'win_rate': self.wins / max(1, self.total_games)
        }


    def replay(self, strategy, delay=2):
        """Rejoue une séquence d'actions pas à pas.
        
        Args:
            strategy (list): Liste de 2 actions [action_round1, action_round2]
                           Chaque action doit être 0 (Pierre), 1 (Feuille) ou 2 (Ciseaux)
            delay (int): Délai en secondes entre chaque action (par défaut 2s)
        """
        import time
        
        if len(strategy) != 2:
            raise ValueError("La stratégie doit contenir exactement 2 actions")
            
        actions_emoji = {
            0: "🪨  Pierre",
            1: "📄  Feuille",
            2: "✂️  Ciseaux"
        }
            
        # Réinitialisation
        state = self.reset()
        total_reward = 0
        print("\n🎯 Début de la partie")
        self.visualisation()
        time.sleep(delay)
        
        # Premier round
        print("\n🎯 Round 1/2")
        state, reward, done, info = self.step(strategy[0])
        total_reward += reward
        print(f"\nVous avez joué : {actions_emoji[info['agent_action']]}")
        print(f"L'adversaire a joué : {actions_emoji[info['opponent_action']]}")
        print(f"Récompense : {reward:+.1f}")
        self.visualisation()
        time.sleep(delay)
        
        # Deuxième round
        print("\n🎯 Round 2/2")
        print("Rappel : L'adversaire va jouer votre coup du round 1 !")
        state, reward, done, info = self.step(strategy[1])
        total_reward += reward
        print(f"\nVous avez joué : {actions_emoji[info['agent_action']]}")
        print(f"L'adversaire a joué : {actions_emoji[info['opponent_action']]}")
        print(f"Récompense : {reward:+.1f}")
        self.visualisation()
        
        print(f"\n🏁 Partie terminée ! Score total : {total_reward:+.1f}")
        return total_reward


# Création d'un fichier __init__.py pour faire du dossier game un module Python
def __init__():
    pass 
