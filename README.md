# Projet de Reinforcement Learning

## Description

Ce projet implémente et compare différents algorithmes de reinforcement learning sur des environnements de jeux personnalisés. Il permet d'analyser automatiquement les performances de chaque algorithme et de déterminer le meilleur modèle pour chaque environnement.

## Structure du Projet

```
├── game/                    # Environnements de jeux
│   ├── __init__.py         # Module d'initialisation
│   └── environments.py     # Tous les environnements (LineWorld, GridWorld, MontyHall 1&2)
├── src/                    # Code source des algorithmes
│   ├── algorithms/         # Implémentations des algorithmes RL
│   │   ├── dp.py          # Dynamic Programming (Policy/Value Iteration)
│   │   ├── monte_carlo.py # Monte Carlo Methods
│   │   ├── td.py          # Temporal Difference (Sarsa, Q-Learning, Expected Sarsa)
│   │   └── dyna.py        # Planning (Dyna-Q, Dyna-Q+)
│   └── utils/             # Utilitaires
├── notebooks/             # Analyse des performances
│   └── analysis.ipynb     # Notebook d'analyse comparative automatique
├── models/                # Modèles sauvegardés
└── requirements.txt       # Dépendances
```

## Algorithmes Implémentés

### Dynamic Programming
- **Policy Iteration** : Amélioration itérative de la politique
- **Value Iteration** : Calcul direct de la fonction valeur optimale

### Monte Carlo Methods
- **Monte Carlo ES** : Exploring Starts
- **On-Policy Monte Carlo** : Apprentissage on-policy
- **Off-Policy Monte Carlo** : Apprentissage off-policy

### Temporal Difference
- **Sarsa** : State-Action-Reward-State-Action (on-policy)
- **Q-Learning** : Off-policy TD control
- **Expected Sarsa** : Utilise l'espérance des valeurs Q

### Planning
- **Dyna-Q** : Combine apprentissage direct et planification
- **Dyna-Q+** : Dyna-Q avec bonus d'exploration

## Environnements de Jeux

### LineWorld
- **Description** : Environnement linéaire avec navigation gauche/droite
- **Objectif** : Atteindre une extrémité de la ligne
- **Complexité** : Simple, idéal pour tester les algorithmes de base

### GridWorld
- **Description** : Grille 2D avec navigation dans 4 directions
- **Objectif** : Atteindre une des sorties (coins)
- **Complexité** : Modérée, test de navigation spatiale

### MontyHall Paradox 1
- **Description** : Problème de Monty Hall classique (3 portes)
- **Objectif** : Maximiser les chances de gagner
- **Complexité** : Décision séquentielle, stratégie optimale non-intuitive

### MontyHall Paradox 2
- **Description** : Problème de Monty Hall étendu (5 portes)
- **Objectif** : Maximiser les chances de gagner
- **Complexité** : Élevée, décisions multiples

## Installation

1. **Cloner le repository**
   ```bash
   git clone [url-du-repo]
   cd [nom-du-repo]
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Lancer l'analyse**
   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```

## Utilisation

### Analyse Automatique des Performances

Le notebook `notebooks/analysis.ipynb` permet de :

1. **Comparer automatiquement** tous les algorithmes sur tous les environnements
2. **Visualiser les performances** avec des graphiques et heatmaps
3. **Obtenir des recommandations** pour chaque environnement
4. **Sauvegarder les résultats** en format CSV

### Exemple d'utilisation des environnements

```python
from game.environments import LineWorld, GridWorld, MontyHallParadox1, MontyHallParadox2

# Créer un environnement
env = LineWorld(length=8)

# Réinitialiser
state = env.reset()

# Prendre une action
next_state, reward, done, info = env.step(action=1)

# Visualiser
env.visualisation()
```

### Exemple d'utilisation des algorithmes

```python
from src.algorithms.td import QLearning
from game.environments import GridWorld

# Créer l'environnement et l'algorithme
env = GridWorld(n_rows=5, n_cols=5)
algorithm = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1)

# Entraîner
history = algorithm.train(num_episodes=1000)

# Évaluer
results = algorithm.evaluate(num_episodes=100)
```

## Métriques d'Évaluation

Le système évalue les algorithmes selon plusieurs critères :

- **Récompense moyenne** : Performance générale
- **Taux de succès** : Pourcentage d'épisodes réussis
- **Temps d'entraînement** : Efficacité computationnelle
- **Nombre d'étapes** : Efficacité de la politique apprise

## Résultats et Analyse

Le notebook d'analyse génère automatiquement :

### Visualisations
- Courbes d'apprentissage par algorithme
- Heatmaps des performances
- Graphiques comparatifs

### Recommandations
- Meilleur algorithme par environnement
- Algorithme le plus polyvalent
- Algorithme le plus rapide
- Algorithme avec le meilleur taux de succès

### Fichiers de sortie
- `results_comparison.csv` : Tableau comparatif détaillé
- Graphiques sauvegardés
- Métriques de performance

## Personnalisation

### Ajouter un nouvel algorithme
1. Créer une nouvelle classe dans `src/algorithms/`
2. Implémenter les méthodes `train()` et `evaluate()`
3. Ajouter à la liste des algorithmes dans le notebook d'analyse

### Ajouter un nouvel environnement
1. Créer une nouvelle classe dans `game/environments.py`
2. Hériter de `BaseEnvironment`
3. Implémenter les méthodes requises
4. Ajouter à la liste des environnements dans le notebook d'analyse

## Contribution

Ce projet est développé dans le cadre d'un projet de 2ème année en intelligence artificielle.

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

**🎯 Projet de Reinforcement Learning - Analyse Comparative Automatique**  
*Implémentation et comparaison d'algorithmes RL sur environnements de jeux* 
