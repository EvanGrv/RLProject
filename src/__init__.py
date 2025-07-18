"""
Package de Reinforcement Learning.

Ce package contient les implémentations des principaux algorithmes de RL :
- Dynamic Programming (dp.py)
- Monte Carlo (monte_carlo.py)
- Temporal Difference (td.py)
- Planning (dyna.py)
- Utilitaires I/O (utils_io.py)
"""

from .dp import PolicyIteration, ValueIteration
from .monte_carlo import MonteCarloES, OnPolicyMC, OffPolicyMC
from .td import Sarsa, QLearning, ExpectedSarsa
from .dyna import DynaQ, DynaQPlus
from .utils_io import save_model, load_model, get_model_info

__version__ = "1.0.0"
__author__ = "Votre nom"

__all__ = [
    # Dynamic Programming
    'PolicyIteration',
    'ValueIteration',
    
    # Monte Carlo
    'MonteCarloES',
    'OnPolicyMC',
    'OffPolicyMC',
    
    # Temporal Difference
    'Sarsa',
    'QLearning',
    'ExpectedSarsa',
    
    # Planning
    'DynaQ',
    'DynaQPlus',
    
    # Utilities
    'save_model',
    'load_model',
    'list_saved_models',
    'get_model_info'
] 