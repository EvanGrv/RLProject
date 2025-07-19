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
        
        # 1. Calcul du prochain état
        if action == 0:  # Gauche
            next_state = max(0, self.state - 1)
        else:  # Droite
            next_state = min(self.length-1, self.state + 1)

        # 2. Calcul de la récompense selon la logique de l'exemple
        if next_state in self.terminals:  # État terminal atteint
            if next_state == 0:  # Terminal gauche
                reward = -1.0
            else:  # Terminal droit (length-1)
                reward = 1.0
        else:  # Déplacement normal
            reward = 0.0
            
        done = next_state in self.terminals
        self.state = next_state
        
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
            if s in self.terminals:
                # Les états terminaux n'ont pas de transitions
                continue
            
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
            # Action gauche (0)
            next_s_left = max(0, s-1)
            if next_s_left == 0:  # Si on atteint le terminal gauche
                R[s, 0] = -1.0
            
            # Action droite (1)
            next_s_right = min(self.length-1, s+1)
            if next_s_right == self.length-1:  # Si on atteint le terminal droit
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


class GridWorld(BaseEnvironment):
    """
    Grid World Environment
    
    L'agent doit se déplacer sur une grille et collecter des récompenses.
    L'agent peut se déplacer vers le haut, le bas, la gauche ou la droite.
    """
    
    def __init__(self, n_rows=5, n_cols=5):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_states = n_rows * n_cols

        # Métriques
        self.steps_count = 0
        self.total_reward = 0
        self.visited_states = set()
        
        # Définition des actions : 0: Gauche, 1: Droite, 2: Haut, 3: Bas
        self.actions = [0, 1, 2, 3]
        
        # Les états terminaux sont dans les coins supérieur droit et inférieur droit de la grille
        self.terminals = [0, self.n_states - 1]
        
        # Définissons les récompenses
        self.rewards = {
            'movement': -1.0,  # Déplacement normal
            'middle': 0.0,    # Vers le milieu
            'no_move': 0.0    # Pas de mouvement possible
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
        """Réinitialisation de l'environnement"""
        self.current_state = self.start_state
        self.done = False
        return self.current_state

    def step(self, action, for_dp=False):
        """Exécute une action et retourne le résultat"""
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
        """Convertit une position (row, col) en numéro d'état"""
        return row * self.n_cols + col

    def _state_to_pos(self, state):
        """Convertit un numéro d'état en position (row, col)"""
        return divmod(state, self.n_cols)

    def _build_transition_matrix(self):
        """Construit la matrice de transition P(s'|s,a)"""
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
        """Construit la matrice des récompenses R(s,a)"""
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
        """Retourne les informations du MDP pour les algorithmes de Dynamic Programming"""
        return {
            'states': range(self.n_states),
            'actions': self.actions,
            'transition_matrix': self._transition_matrix,
            'reward_matrix': self._reward_matrix,
            'terminals': self.terminals,
            'gamma': 0.99
        }

    def visualisation(self):
        """Affiche l'état actuel de l'environnement"""
        print("\n" + "="*40)
        print("Grid World".center(40))
        print("-"*40)
        
        grid = [['[ ]' for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        
        # Marquer les terminaux
        for terminal in self.terminals:
            row, col = self._state_to_pos(terminal)
            grid[row][col] = '[T]'
        
        # Position actuelle
        row, col = self._state_to_pos(self.current_state)
        grid[row][col] = '[A]'
        
        # Afficher la grille avec bordures
        print('+' + '---+'*self.n_cols)
        for row in grid:
            print('|' + '|'.join(row) + '|')
            print('+' + '---+'*self.n_cols)
        
        print(f"\nPosition: {self.current_state} ({row},{col})")
        print("="*40)

    def jeu_manuel(self):
        """Permet à un utilisateur de jouer contre l'environnement"""
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
        """Retourne des métriques pour connaître les performances de l'agent"""
        current_row, current_col = self._state_to_pos(self.current_state)
        return {
            'current_state': self.current_state,
            'steps_taken': self.steps_count,
            'total_reward': self.total_reward,
            'visited_states': self.visited_states,
            'average_reward': self.total_reward / max(1, self.steps_count)
        }

    def reset_metrics(self):
        """Réinitialise les métriques pour une nouvelle expérience"""
        self.steps_count = 0
        self.total_reward = 0
        self.visited_states = set()
    
    def replay(self, strategy, delay=1):
        """Rejoue une séquence d'actions pas à pas"""
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


class MontyHallParadox1(BaseEnvironment):
    """
    Monty Hall Paradox 1
    
    L'agent est un candidat au jeu Monty Hall, Il doit prendre 2 décisions successives.
    Dans cet environnement, il y a 3 portes A, B et C.
    Au démarrage de l'environnement, une porte est tirée au hasard de manière cachée pour l'agent.
    Il s'agit de la porte gagnante.
    """
    
    def __init__(self):
        # Configuration du jeu
        self.doors = [0, 1, 2]  # Portes A, B, C
        self.R = [0.0, 1.0]  # Récompenses possibles
        self.A = [0, 1]  # 0: rester, 1: changer
        
        # État courant
        self.current_state = None
        self.winning_door = None
        self.revealed_door = None
        self.first_choice = None
        
        # Métriques de performance
        self.total_games = 0
        self.wins = 0
        self.stay_wins = 0
        self.switch_wins = 0
        self.total_reward = 0
        
        # Construction des matrices pour DP
        self.states, self.state_to_idx = self._build_states()
        self.transition_matrix = self._build_transition_matrix()
        self.reward_matrix = self._build_reward_matrix()

    def reset(self):
        """Réinitialise l'environnement pour un nouvel épisode"""
        self.winning_door = random.choice(self.doors)
        self.revealed_door = None
        self.first_choice = None
        self.current_state = (self.winning_door, None, None)
        return self.current_state

    def step(self, action, is_dynamic_programming=False):
        """Exécute une action dans l'environnement"""
        if self.current_state == (None, None, None):
            return self.current_state, 0.0, True, {}
            
        winning, chosen, revealed = self.current_state
        
        # Phase 1: Premier choix de porte
        if chosen is None:
            if action not in self.doors:
                raise ValueError(f"Action invalide {action}. Doit être entre 0 et 2.")
                
            # Révéler une porte non gagnante différente du choix
            # Dans Monty Hall, l'hôte révèle toujours une porte perdante qui n'est pas celle choisie
            available_doors = [d for d in self.doors if d != action and d != winning]
            
            # Si le joueur a choisi la porte gagnante, l'hôte peut révéler n'importe laquelle des 2 autres
            if action == winning:
                available_doors = [d for d in self.doors if d != action]
            
            # S'assurer qu'il y a au moins une porte à révéler
            if not available_doors:
                # Cas de sécurité : révéler une porte différente de celle choisie
                available_doors = [d for d in self.doors if d != action]
            
            revealed = random.choice(available_doors)
            
            self.first_choice = action
            self.revealed_door = revealed
            next_state = (winning, action, revealed)
            
            if is_dynamic_programming:
                return next_state, 0.0, False, {}
            
            self.current_state = next_state
            return next_state, 0.0, False, {
                'phase': 1,
                'door_chosen': ['A', 'B', 'C'][action],
                'door_revealed': ['A', 'B', 'C'][revealed]
            }
            
        # Phase 2: Décision de rester ou changer
        if action not in self.A:
            raise ValueError(f"Action invalide {action}. Doit être 0 (rester) ou 1 (changer).")
            
        remaining = [d for d in self.doors if d != chosen and d != revealed][0]
        final_choice = chosen if action == 0 else remaining
        reward = 1.0 if final_choice == winning else 0.0
        
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
        
        next_state = (None, None, None)
        self.current_state = next_state
        
        if is_dynamic_programming:
            return next_state, reward, True, {}
            
        return next_state, reward, True, {
            'phase': 2,
            'final_choice': ['A', 'B', 'C'][final_choice],
            'winning_door': ['A', 'B', 'C'][winning],
            'action': 'resté' if action == 0 else 'changé'
        }

    def _build_states(self):
        """Construit la liste des états et leur mapping vers des indices"""
        states = []
        state_to_idx = {}
        # Phase 1 (3 états)
        for winning in self.doors:
            state = (winning, None, None)
            states.append(state)
            state_to_idx[state] = len(states) - 1
        
        # Phase 2 (9 états)
        for winning in self.doors:
            for chosen in self.doors:
                if chosen == winning:
                    for revealed in [d for d in self.doors if d != chosen]:
                        state = (winning, chosen, revealed)
                        states.append(state)
                        state_to_idx[state] = len(states) - 1
                else:
                    revealed = [d for d in self.doors if d != chosen and d != winning][0]
                    state = (winning, chosen, revealed)
                    states.append(state)
                    state_to_idx[state] = len(states) - 1
        
        # État terminal
        terminal_state = (None, None, None)
        states.append(terminal_state)
        state_to_idx[terminal_state] = len(states) - 1
        
        return states, state_to_idx

    def _build_transition_matrix(self):
        """Construit la matrice de transition P(s'|s,a)"""
        n_states = len(self.states)
        n_actions = max(len(self.doors), len(self.A))
        P = np.zeros((n_states, n_actions, n_states))
        
        # Remplir la matrice
        for s_idx, state in enumerate(self.states[:-1]):  # Exclure l'état terminal
            winning, chosen, revealed = state
            
            if chosen is None:  # Phase 1
                for action in self.doors:
                    next_state, _, _ = self.step(action, is_dynamic_programming=True)
                    next_idx = self.state_to_idx[next_state]
                    
                    if action == winning:
                        P[s_idx, action, next_idx] = 0.5
                    else:
                        P[s_idx, action, next_idx] = 1.0
            else:  # Phase 2
                terminal_idx = self.state_to_idx[(None, None, None)]
                for action in self.A:
                    P[s_idx, action, terminal_idx] = 1.0
        
        return P

    def _build_reward_matrix(self):
        """Construit la matrice de récompense R(s,a)"""
        n_states = len(self.states)
        n_actions = max(len(self.doors), len(self.A))
        R = np.zeros((n_states, n_actions))
        
        # Les récompenses ne sont données qu'en phase 2
        for s in range(n_states):
            if s < 3:  # États de phase 1
                continue
                
            state = self.states[s]
            if state == (None, None, None):  # État terminal
                continue
                
            winning, chosen, revealed = state
            remaining = [d for d in self.doors if d != chosen and d != revealed][0]
            
            # Action 0 (rester)
            R[s, 0] = 1.0 if chosen == winning else 0.0
            
            # Action 1 (changer)
            R[s, 1] = 1.0 if remaining == winning else 0.0
        
        return R

    def get_mdp_info(self):
        """Retourne les informations du MDP pour les algorithmes de DP"""
        return {
            'states': range(len(self.transition_matrix)),
            'actions': self.A,
            'transition_matrix': self.transition_matrix,
            'reward_matrix': self.reward_matrix,
            'terminals': [len(self.transition_matrix) - 1],  # Dernier état = terminal
            'gamma': 0.99
        }

    def visualisation(self):
        """Affiche l'état actuel du jeu"""
        print("\n" + "="*50)
        print("🚪 MONTY HALL 🚪".center(50))
        print("-"*50)
        
        if self.current_state == (None, None, None):
            print("Partie terminée !".center(50))
            print(f"La porte gagnante était : {['A', 'B', 'C'][self.winning_door]}")
            return
            
        doors = ['[A]', '[B]', '[C]']
        
        if self.first_choice is None:
            print("Choisissez une porte :".center(50))
            print(" ".join(doors).center(50))
        else:
            doors[self.revealed_door] = '[X]'  # Porte révélée
            doors[self.first_choice] = '[*]'  # Porte choisie
            print("X : Porte révélée (vide)".center(50))
            print("* : Votre choix initial".center(50))
            print(" ".join(doors).center(50))
            print("\nVoulez-vous :")
            print("0 : Rester sur votre choix")
            print("1 : Changer de porte")
            
        print("="*50)

    def jeu_manuel(self):
        """Permet à un utilisateur de jouer une partie"""
        self.reset()
        print("\n=== Monty Hall ===")
        print("Bienvenue dans le paradoxe de Monty Hall!")
        print("Il y a trois portes : A(0), B(1) et C(2)")
        print("Derrière l'une d'elles se trouve le grand prix!")
        
        while True:
            self.visualisation()
            
            if self.first_choice is None:
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
        """Retourne les métriques de performance"""
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
        """Rejoue une séquence d'actions pas à pas"""
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


class MontyHallParadox2(BaseEnvironment):
    """
    Monty Hall Paradox 2
    
    Il s'agit du même environnement que précédemment, seulement maintenant 5 portes sont 
    disponibles, et l'agent doit effectuer 4 actions successives avant l'ouverture d'une 
    des deux portes restantes.
    """
    
    def __init__(self):
        # Configuration du jeu
        self.doors = [0, 1, 2, 3, 4]  # 5 Portes A, B, C, D, E
        self.R = [0.0, 1.0]  # Récompenses possibles
        self.A = [0, 1]  # 0: rester, 1: changer
        
        # État courant
        self.current_state = None
        self.winning_door = None
        self.revealed_doors = []  # Liste des portes révélées
        self.current_choice = None  # Porte actuellement choisie
        self.phase = 0  # Phase du jeu (0-4)
        
        # Métriques de performance
        self.total_games = 0
        self.wins = 0
        self.stay_wins = 0
        self.switch_wins = 0
        self.total_reward = 0
        
        # Construction des états et de leur mapping
        self.states, self.state_to_idx = self._build_states()
        
        # Construction des matrices pour DP
        self._transition_matrix = self._build_transition_matrix()
        self._reward_matrix = self._build_reward_matrix()

    def reset(self):
        """Réinitialise l'environnement pour un nouvel épisode"""
        self.winning_door = random.choice(self.doors)
        self.revealed_doors = []
        self.current_choice = None
        self.phase = 0
        self.current_state = (self.winning_door, None, tuple())  # winning, current_choice, revealed_doors
        return self.current_state

    def _verify_remaining_doors(self):
        """Vérifie qu'il reste bien 2 portes à la fin du jeu"""
        if self.phase == 3:  # Phase finale
            revealed = self.current_state[2]
            remaining = [d for d in self.doors if d not in revealed and d != self.current_choice]
            if len(remaining) != 1:
                raise ValueError(f"Il devrait rester exactement 1 porte non révélée en plus de la porte choisie. Actuellement : {len(remaining)}")
            return True
        return False

    def step(self, action, is_dynamic_programming=False):
        """Exécute une action dans l'environnement"""
        if self.current_state == (None, None, None):
            return self.current_state, 0.0, True, {}
            
        winning, chosen, revealed = self.current_state
        revealed = list(revealed) if isinstance(revealed, tuple) else revealed

        # Phase finale (4ème action)
        if self.phase == 3:
            self._verify_remaining_doors()
            if action not in self.A:
                raise ValueError(f"Phase finale : l'action doit être 0 (rester) ou 1 (changer), pas {action}")
            
            available = [d for d in self.doors if d not in revealed and d != chosen]
            final_choice = chosen if action == 0 else available[0]
            reward = 1.0 if final_choice == winning else 0.0
            
            if not is_dynamic_programming:
                self.total_games += 1
                self.total_reward += reward
                if reward == 1.0:
                    self.wins += 1
                    if action == 0:
                        self.stay_wins += 1
                    else:
                        self.switch_wins += 1
            
            next_state = (None, None, None)
            self.current_state = next_state
            
            if is_dynamic_programming:
                return next_state, reward, True
                
            return next_state, reward, True, {
                'phase': self.phase + 1,
                'final_choice': ['A', 'B', 'C', 'D', 'E'][final_choice],
                'winning_door': ['A', 'B', 'C', 'D', 'E'][winning],
                'action': 'resté' if action == 0 else 'changé'
            }
        
        # Phases intermédiaires (1-3)
        if action not in self.doors:
            raise ValueError(f"Phase {self.phase + 1} : l'action doit être une porte (0-4), pas {action}")
        
        if action in revealed:
            raise ValueError(f"Phase {self.phase + 1} : la porte {['A', 'B', 'C', 'D', 'E'][action]} a déjà été révélée!")
            
        # Mise à jour du choix courant
        self.current_choice = action
        
        # Révélation d'une nouvelle porte
        available_doors = [d for d in self.doors if d not in revealed and d != action]
        if len(available_doors) <= 1:
            raise ValueError(f"Phase {self.phase + 1} : plus assez de portes disponibles pour la révélation")
            
        if action == winning:
            # Si l'agent a choisi la porte gagnante, Monty révèle une porte au hasard
            to_reveal = random.choice([d for d in available_doors if d != winning])
        else:
            # Sinon, Monty révèle une porte non gagnante
            to_reveal = random.choice([d for d in available_doors if d != winning])
        
        revealed.append(to_reveal)
        self.phase += 1
        
        next_state = (winning, action, tuple(revealed))
        self.current_state = next_state
        
        if is_dynamic_programming:
            return next_state, 0.0, False
        
        return next_state, 0.0, False, {
            'phase': self.phase,
            'door_chosen': ['A', 'B', 'C', 'D', 'E'][action],
            'door_revealed': ['A', 'B', 'C', 'D', 'E'][to_reveal]
        }

    def _build_states(self):
        """Construit la liste des états et leur mapping vers des indices"""
        states = []
        state_to_idx = {}
        
        # Phase 1 (5 états initiaux, un pour chaque porte gagnante possible)
        for winning in self.doors:
            state = (winning, None, tuple())
            states.append(state)
            state_to_idx[state] = len(states) - 1
        
        # Phase 2 (après premier choix)
        for winning in self.doors:
            for chosen in self.doors:
                for revealed in self.doors:
                    if revealed != chosen and revealed != winning:
                        state = (winning, chosen, tuple([revealed]))
                        states.append(state)
                        state_to_idx[state] = len(states) - 1
        
        # Phase 3 (après deuxième choix)
        for winning in self.doors:
            for chosen in self.doors:
                for revealed1 in self.doors:
                    for revealed2 in self.doors:
                        if (revealed1 != chosen and revealed1 != winning and
                            revealed2 != chosen and revealed2 != winning and
                            revealed1 != revealed2):
                            state = (winning, chosen, tuple([revealed1, revealed2]))
                            states.append(state)
                            state_to_idx[state] = len(states) - 1
        
        # Phase 4 (après troisième choix)
        for winning in self.doors:
            for chosen in self.doors:
                for revealed1 in self.doors:
                    for revealed2 in self.doors:
                        for revealed3 in self.doors:
                            if (revealed1 != chosen and revealed1 != winning and
                                revealed2 != chosen and revealed2 != winning and
                                revealed3 != chosen and revealed3 != winning and
                                revealed1 != revealed2 and revealed1 != revealed3 and
                                revealed2 != revealed3):
                                state = (winning, chosen, tuple([revealed1, revealed2, revealed3]))
                                states.append(state)
                                state_to_idx[state] = len(states) - 1
        
        # État terminal
        terminal_state = (None, None, None)
        states.append(terminal_state)
        state_to_idx[terminal_state] = len(states) - 1
        
        return states, state_to_idx

    def _build_transition_matrix(self):
        """Construit la matrice de transition P(s'|s,a)"""
        n_states = len(self.states)
        n_actions = max(len(self.doors), len(self.A))
        P = np.zeros((n_states, n_actions, n_states))
        
        # Remplir la matrice
        for s_idx, state in enumerate(self.states[:-1]):  # Exclure l'état terminal
            winning, chosen, revealed = state
            revealed = list(revealed) if isinstance(revealed, tuple) else revealed
            
            # Phase 1-3 : Choix des portes
            if len(revealed) < 3:
                for action in self.doors:
                    if action not in revealed and action != chosen:
                        # Calculer les portes disponibles pour la révélation
                        available = [d for d in self.doors if d not in revealed and d != action]
                        if winning in available and action != winning:
                            # Si la porte gagnante est disponible et l'agent n'a pas choisi la bonne porte
                            # Monty révèle une porte non gagnante
                            non_winning = [d for d in available if d != winning]
                            prob = 1.0 / len(non_winning)
                            for to_reveal in non_winning:
                                new_revealed = list(revealed)
                                new_revealed.append(to_reveal)
                                next_state = (winning, action, tuple(new_revealed))
                                next_idx = self.state_to_idx[next_state]
                                P[s_idx, action, next_idx] = prob
                        else:
                            # Si l'agent a choisi la bonne porte ou si elle n'est plus disponible
                            # Monty révèle une porte au hasard parmi les disponibles
                            prob = 1.0 / len(available)
                            for to_reveal in available:
                                new_revealed = list(revealed)
                                new_revealed.append(to_reveal)
                                next_state = (winning, action, tuple(new_revealed))
                                next_idx = self.state_to_idx[next_state]
                                P[s_idx, action, next_idx] = prob
            
            # Phase 4 : Décision finale
            elif len(revealed) == 3:
                terminal_idx = self.state_to_idx[(None, None, None)]
                for action in self.A:  # 0: rester, 1: changer
                    P[s_idx, action, terminal_idx] = 1.0
        
        return P

    def _build_reward_matrix(self):
        """Construit la matrice de récompense R(s,a)"""
        n_states = len(self.states)
        n_actions = max(len(self.doors), len(self.A))
        R = np.zeros((n_states, n_actions))
        
        # Les récompenses ne sont données qu'en phase 4
        for s_idx, state in enumerate(self.states):
            if state == (None, None, None):  # État terminal
                continue
                
            winning, chosen, revealed = state
            if not isinstance(revealed, tuple):
                continue
                
            # Phase 4 : après 3 révélations
            if len(revealed) == 3:
                available = [d for d in self.doors if d not in revealed and d != chosen]
                
                # Action 0 (rester)
                R[s_idx, 0] = 1.0 if chosen == winning else 0.0
                
                # Action 1 (changer)
                R[s_idx, 1] = 1.0 if available[0] == winning else 0.0
        
        return R

    def get_mdp_info(self):
        """Retourne les informations du MDP pour les algorithmes de DP"""
        return {
            'states': range(len(self.transition_matrix)),
            'actions': self.A,
            'transition_matrix': self.transition_matrix,
            'reward_matrix': self.reward_matrix,
            'terminals': [len(self.transition_matrix) - 1],  # Dernier état = terminal
            'gamma': 0.99
        }

    def visualisation(self):
        """Affiche l'état actuel du jeu"""
        print("\n" + "="*50)
        print("🎮 MONTY HALL 2 - 5 PORTES 🎮".center(50))
        print("-"*50)
        
        if self.current_state == (None, None, None):
            print("Partie terminée !".center(50))
            print(f"La porte gagnante était : {['A', 'B', 'C', 'D', 'E'][self.winning_door]}")
            return
            
        doors = ['[A]', '[B]', '[C]', '[D]', '[E]']
        revealed = self.current_state[2]
        
        # Marquer les portes révélées et le choix actuel
        if revealed:
            for r in revealed:
                doors[r] = '[X]'  # Porte révélée
        if self.current_choice is not None:
            doors[self.current_choice] = '[*]'  # Porte choisie
        
        print(f"Phase {self.phase + 1}/4".center(50))
        if self.phase < 3:
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
        """Permet à un utilisateur de jouer une partie"""
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
                if self.phase < 3:
                    action = int(input(f"\nChoisissez une porte (0-4): "))
                    if action not in self.doors:
                        print("Action invalide! Choisissez entre 0 et 4")
                        continue
                    if action in self.current_state[2]:
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
        """Rejoue une séquence d'actions pas à pas"""
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


# Création d'un fichier __init__.py pour faire du dossier game un module Python
def __init__():
    pass 
