# Projet Reinforcement Learning - Algorithmes Classiques

Ce projet implémente les principaux algorithmes de reinforcement learning (RL) en Python, avec une structure modulaire et des outils d'analyse complets.

## 🎯 Objectifs

Implémenter et comparer les algorithmes fondamentaux de RL :
- **Dynamic Programming** : Policy Iteration, Value Iteration
- **Monte Carlo** : ES, On-policy first-visit, Off-policy
- **Temporal Difference** : Sarsa, Q-Learning, Expected Sarsa
- **Planning** : Dyna-Q, Dyna-Q+

## 📁 Structure du Projet

```
DL - Projet 2eme année/
├── src/                    # Code source principal
│   ├── __init__.py        # Imports du package
│   ├── dp.py              # Dynamic Programming
│   ├── monte_carlo.py     # Algorithmes Monte Carlo
│   ├── td.py              # Temporal Difference
│   ├── dyna.py            # Planning (Dyna-Q, Dyna-Q+)
│   └── utils_io.py        # Utilitaires I/O (pickle, JSON, NPZ)
├── models/                 # Modèles sauvegardés
│   └── .gitkeep           # Maintient le dossier dans Git
├── tests/                  # Tests unitaires
│   ├── test_dp.py         # Tests Dynamic Programming
│   ├── test_monte_carlo.py # Tests Monte Carlo
│   ├── test_td.py         # Tests Temporal Difference
│   ├── test_dyna.py       # Tests Planning
│   └── test_utils_io.py   # Tests utilitaires I/O
├── notebooks/              # Notebooks d'analyse
│   └── analysis.ipynb     # Analyse et visualisation
├── .github/               # Configuration CI/CD
│   └── workflows/
│       └── ci.yml         # GitHub Actions
├── requirements.txt       # Dépendances Python
└── README.md             # Documentation (ce fichier)
```

## 🚀 Installation

### 1. Cloner le repository
```bash
git clone <votre-repo-url>
cd "DL - Projet 2eme année"
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## 🎮 Utilisation

### Exemple d'utilisation basique

```python
import numpy as np
from src import QLearning, save_model, load_model

# Créer un environnement (exemple avec gym)
import gym
env = gym.make('FrozenLake-v1')

# Initialiser l'algorithme Q-Learning
q_learning = QLearning(
    env=env,
    alpha=0.1,      # Taux d'apprentissage
    gamma=0.99,     # Facteur de discount
    epsilon=0.1     # Exploration epsilon-greedy
)

# Entraîner le modèle
results = q_learning.train(num_episodes=1000)

# Sauvegarder le modèle
q_learning.save('models/q_learning_model.pkl')

# Charger un modèle existant
q_learning_loaded = QLearning(env)
q_learning_loaded.load('models/q_learning_model.pkl')
```

### Formats de sauvegarde supportés

```python
from src.utils_io import save_model, load_model

# Différents formats disponibles
save_model(data, 'models/model.pkl', format='pickle')    # Par défaut
save_model(data, 'models/model.json', format='json')     # JSON
save_model(data, 'models/model.joblib', format='joblib') # Joblib
save_model(data, 'models/model.npz', format='npz')       # NumPy NPZ

# Chargement automatique basé sur l'extension
data = load_model('models/model.pkl', format='auto')
```

## 🧪 Tests

Exécuter tous les tests :
```bash
pytest tests/
```

Exécuter un module spécifique :
```bash
pytest tests/test_dp.py -v
```

Tests avec couverture :
```bash
pytest --cov=src tests/
```

## 📊 Analyse des Résultats

Utilisez le notebook `notebooks/analysis.ipynb` pour :
- Charger et explorer les modèles sauvegardés
- Visualiser les courbes de convergence
- Comparer les performances des algorithmes
- Analyser les résultats d'entraînement

```bash
jupyter notebook notebooks/analysis.ipynb
```

## 🤖 Algorithmes Implémentés

### Dynamic Programming (`src/dp.py`)
- **PolicyIteration** : Itération de politique avec évaluation complète
- **ValueIteration** : Itération de valeur avec mise à jour directe

### Monte Carlo (`src/monte_carlo.py`)
- **MonteCarloES** : Exploring Starts avec tous les couples état-action
- **OnPolicyFirstVisitMC** : On-policy first-visit avec epsilon-greedy
- **OffPolicyMC** : Off-policy avec importance sampling

### Temporal Difference (`src/td.py`)
- **Sarsa** : On-policy TD control
- **QLearning** : Off-policy TD control
- **ExpectedSarsa** : Sarsa avec valeur espérée

### Planning (`src/dyna.py`)
- **DynaQ** : Q-Learning avec planification intégrée
- **DynaQPlus** : Dyna-Q avec bonus d'exploration

## ⚙️ Paramètres Principaux

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| `alpha` | Taux d'apprentissage | 0.1 |
| `gamma` | Facteur de discount | 0.9 |
| `epsilon` | Exploration epsilon-greedy | 0.1 |
| `theta` | Seuil de convergence (DP) | 1e-6 |
| `n_planning` | Étapes de planification (Dyna) | 5 |
| `kappa` | Bonus d'exploration (Dyna-Q+) | 0.001 |

## 📈 Exemple d'Entraînement Complet

```python
import gym
from src import QLearning, DynaQ, save_model
import matplotlib.pyplot as plt

# Environnement
env = gym.make('FrozenLake-v1')

# Algorithmes à comparer
algorithms = {
    'Q-Learning': QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1),
    'Dyna-Q': DynaQ(env, alpha=0.1, gamma=0.99, epsilon=0.1, n_planning=5)
}

# Entraîner et sauvegarder
results = {}
for name, algo in algorithms.items():
    print(f"Entraînement {name}...")
    results[name] = algo.train(num_episodes=1000)
    algo.save(f'models/{name.lower().replace("-", "_")}.pkl')

# Visualiser les résultats
for name, result in results.items():
    if result and 'history' in result:
        rewards = [entry['reward'] for entry in result['history']]
        plt.plot(rewards, label=name)

plt.xlabel('Épisode')
plt.ylabel('Récompense')
plt.title('Comparaison des Algorithmes')
plt.legend()
plt.show()
```

## 🔧 Développement

### Ajouter un nouvel algorithme

1. Créer la classe dans le module approprié
2. Hériter de la structure commune avec méthodes `train()`, `save()`, `load()`
3. Ajouter les imports dans `src/__init__.py`
4. Créer les tests dans `tests/`
5. Documenter dans le README

### Structure d'une classe d'algorithme

```python
class NouvelAlgorithme:
    def __init__(self, env, **params):
        self.env = env
        self.history = []
        # Autres paramètres...
    
    def train(self, num_episodes=1000):
        # Logique d'entraînement
        return {'history': self.history}
    
    def save(self, filepath):
        model_data = {
            'param1': self.param1,
            'history': self.history
        }
        save_model(model_data, filepath)
    
    def load(self, filepath):
        model_data = load_model(filepath)
        self.param1 = model_data['param1']
        self.history = model_data['history']
```

## 📚 Ressources

- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Gym Documentation](https://gym.openai.com/)
- [NumPy Documentation](https://numpy.org/doc/)

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit les changements (`git commit -m 'Ajout nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## 🐛 Problèmes Connus

- Les algorithmes DP nécessitent un environnement avec dynamique connue
- Les formats JSON ne supportent pas tous les types NumPy complexes
- Joblib requis pour le format joblib (`pip install joblib`)

## 📝 Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## 👥 Auteurs

- **Votre nom** - Développement initial
- **Équipe** - Contributions diverses

## 🙏 Remerciements

- Professeurs et encadrants du cours de Deep Learning
- Communauté OpenAI Gym
- Sutton & Barto pour leur livre référence 