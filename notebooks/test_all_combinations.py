import sys
import os
from typing import Any, Dict

# Ajout des chemins pour l'import dynamique
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../game')))

from game import environments
from itertools import combinations
import importlib

# Liste des environnements RL disponibles
env_classes = {
    'LineWorld': environments.LineWorld,
    'GridWorld': environments.GridWorld,
    'MontyHallParadox1': environments.MontyHallParadox1,
    'MontyHallParadox2': environments.MontyHallParadox2,
    'RockPaperScissors': environments.RockPaperScissors
}

# Liste des modèles RL disponibles
model_classes = {
    'PolicyIteration': ('dp', 'PolicyIteration'),
    'ValueIteration': ('dp', 'ValueIteration'),
    'Sarsa': ('td', 'Sarsa'),
    'QLearning': ('td', 'QLearning'),
    'ExpectedSarsa': ('td', 'ExpectedSarsa'),
    'MonteCarloES': ('monte_carlo', 'MonteCarloES'),
    'OnPolicyMC': ('monte_carlo', 'OnPolicyMC'),
    'OffPolicyMC': ('monte_carlo', 'OffPolicyMC'),
    'DynaQ': ('dyna', 'DynaQ'),
    'DynaQPlus': ('dyna', 'DynaQPlus')
}

def get_model_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

# Hyperparamètres simples pour chaque modèle
simple_hyperparams = {
    'PolicyIteration': dict(gamma=0.9, theta=1e-4),
    'ValueIteration': dict(gamma=0.9, theta=1e-4),
    'Sarsa': dict(gamma=0.9, alpha=0.1, epsilon=0.1, num_episodes=10),
    'QLearning': dict(gamma=0.9, alpha=0.1, epsilon=0.1, num_episodes=10),
    'ExpectedSarsa': dict(gamma=0.9, alpha=0.1, epsilon=0.1, num_episodes=10),
    'MonteCarloES': dict(gamma=0.9, num_episodes=10),  # pas d'epsilon
    'OnPolicyMC': dict(gamma=0.9, epsilon=0.1, num_episodes=10),
    'OffPolicyMC': dict(gamma=0.9, epsilon=0.1, num_episodes=10),
    'DynaQ': dict(gamma=0.9, alpha=0.1, epsilon=0.1, num_episodes=10),
    'DynaQPlus': dict(gamma=0.9, alpha=0.1, epsilon=0.1, num_episodes=10),
}

# Wrapper pour forcer l'utilisation des indices d'état
class IndexEnvAdapter:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    def reset(self):
        return self.env.reset(as_index=True)
    def step(self, action):
        return self.env.step(action, as_index=True)
    def get_state_space(self):
        return range(self.observation_space.n)
    def get_action_space(self):
        return range(self.action_space.n)

print("\n=== TEST AUTOMATIQUE DE TOUTES LES COMBINAISONS MODÈLE/ENVIRONNEMENT ===\n")

for env_name, env_class in env_classes.items():
    for model_name, (module_name, class_name) in model_classes.items():
        print(f"\n--- Test {model_name} sur {env_name} ---")
        try:
            env = env_class()
            # Patch observation_space/action_space si besoin
            if not hasattr(env, 'observation_space') and hasattr(env, 'get_state_space'):
                states = list(env.get_state_space())
                env.observation_space = type('obj', (), {'n': len(states)})()
            if not hasattr(env, 'action_space') and hasattr(env, 'get_action_space'):
                actions = list(env.get_action_space())
                env.action_space = type('obj', (), {'n': len(actions)})()
            # Utilise le wrapper si besoin
            if hasattr(env, 'state_to_index') and model_name in ['Sarsa', 'QLearning', 'ExpectedSarsa', 'DynaQ', 'DynaQPlus', 'OnPolicyMC', 'OffPolicyMC']:
                env = IndexEnvAdapter(env)
            ModelClass = get_model_class(module_name, class_name)
            params = simple_hyperparams.get(model_name, {})
            if model_name in ['PolicyIteration', 'ValueIteration']:
                model = ModelClass(env, gamma=params['gamma'], theta=params['theta'])
                result = model.train(max_iterations=10)
            elif model_name in ['Sarsa', 'QLearning', 'ExpectedSarsa', 'DynaQ', 'DynaQPlus']:
                model = ModelClass(env, gamma=params['gamma'], alpha=params['alpha'], epsilon=params['epsilon'])
                result = model.train(num_episodes=params['num_episodes'])
            elif model_name in ['MonteCarloES']:
                model = ModelClass(env, gamma=params['gamma'])
                result = model.train(num_episodes=params['num_episodes'])
            elif model_name in ['OnPolicyMC', 'OffPolicyMC']:
                model = ModelClass(env, gamma=params['gamma'], epsilon=params['epsilon'])
                result = model.train(num_episodes=params['num_episodes'])
            else:
                model = ModelClass(env)
                result = model.train()
            print("  ✅ Succès")
            if hasattr(model, 'evaluate'):
                try:
                    eval_result = model.evaluate(num_episodes=5)
                    print(f"  Score: {eval_result}")
                except Exception as e:
                    print(f"  ⚠️ Erreur à l'évaluation: {e}")
        except Exception as e:
            print(f"  ❌ Échec: {e}") 