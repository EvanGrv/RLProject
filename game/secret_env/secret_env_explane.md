# 🕵️ Analyse Monte Carlo sur les Environnements Secrets

Ce dossier contient une analyse complète des algorithmes Monte Carlo appliqués aux environnements secrets fournis via des bibliothèques natives compilées.

## 📁 Structure du Dossier

```
game/secret_env/
├── monte_carlo_analysis.ipynb    # Notebook principal d'analyse
├── secret_envs_wrapper.py        # Wrapper Python pour les environnements secrets
├── README.md                     # Ce fichier
└── libs/                         # Bibliothèques natives compilées
    ├── secret_envs.dll          # Windows
    ├── libsecret_envs.so        # Linux  
    ├── libsecret_envs.dylib     # macOS ARM
    └── libsecret_envs_intel_macos.dylib  # macOS Intel
```

## 🎯 Objectifs de l'Analyse

Le notebook `monte_carlo_analysis.ipynb` implémente et compare **3 algorithmes Monte Carlo** sur **4 environnements secrets** :

### 📋 Algorithmes Testés
1. **Monte Carlo Exploring Starts (MC-ES)**
   - Garantit l'exploration avec départs aléatoires
   - Converge vers la politique optimale sous certaines conditions

2. **On-Policy Monte Carlo**
   - Politique ε-greedy avec décroissance d'epsilon
   - Apprend la politique qu'il suit

3. **Off-Policy Monte Carlo**
   - Importance sampling pour séparer exploration et exploitation
   - Politique cible déterministe, politique de comportement ε-greedy

### 🌟 Environnements Secrets
- **SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3**
- Interface native via ctypes
- Actions disponibles dynamiques selon l'état
- Récompenses basées sur des scores différentiels

## 🚀 Utilisation

### Prérequis
```bash
pip install numpy pandas matplotlib seaborn tqdm
```

### Lancement
1. Ouvrez `monte_carlo_analysis.ipynb` dans Jupyter Lab/Notebook
2. Exécutez les cellules dans l'ordre séquentiel
3. Ajustez le paramètre `EPISODES` selon vos besoins de temps

### Paramètres Configurables
```python
EPISODES = 800          # Nombre d'épisodes par algorithme
gamma = 0.99           # Facteur de discount
epsilon = 0.3-0.4      # Taux d'exploration initial
```

## 📊 Analyses Fournies

### 1. 📈 Courbes d'Apprentissage
- **Récompenses par épisode** avec moyenne mobile
- **Évolution des Q-values** moyennes
- **Longueur des épisodes** au fil du temps
- **Analyse de stabilité** (écart-type des récompenses)

### 2. 🏆 Comparaisons Globales  
- **Heatmap des performances** par algorithme/environnement
- **Taux de succès** comparatifs
- **Q-values finales** moyennes
- **Scores composites** globaux

### 3. 📋 Métriques Détaillées
- Taux de succès des épisodes
- Récompenses moyennes et finales
- Stabilité d'apprentissage
- Caractéristiques spécifiques (ε, importance sampling weights)

### 4. 🎯 Recommandations
- Meilleur algorithme par environnement
- Performance globale des algorithmes
- Suggestions d'hyperparamètres

## 🔧 Architecture Technique

### Adaptateur d'Environnement
```python
class SecretEnvAdapter:
    - Traduit l'API native vers l'API Gym standard
    - Gère les actions disponibles dynamiques  
    - Calcule les récompenses différentielles
    - Limite les épisodes infinis
```

### Algorithmes Monte Carlo
```python
class SecretMonteCarloES:        # Exploring Starts
class SecretOnPolicyMC:          # ε-greedy On-Policy  
class SecretOffPolicyMC:         # Importance Sampling
```

### Fonctions de Visualisation
```python
plot_learning_curves()          # Courbes d'apprentissage
plot_performance_comparison()   # Comparaisons globales
analyze_algorithm_characteristics()  # Analyses détaillées
```

## 📈 Outputs Générés

### Fichiers de Sortie
- `secret_env_monte_carlo_results.csv` - Tableau récapitulatif exportable

### Visualisations
- **16 graphiques** de courbes d'apprentissage (4 par environnement)
- **4 graphiques** de comparaison globale
- **Analyses textuelles** détaillées
- **Recommandations** personnalisées

## 🐛 Résolution de Problèmes

### Erreurs Communes
```bash
# Erreur d'import des environnements
❌ ImportError: DLL load failed
✅ Vérifiez que les libs natives correspondent à votre OS

# Performance lente  
❌ Entraînement trop long
✅ Réduisez EPISODES à 200-400 pour des tests rapides

# Actions invalides
❌ Action not in available_actions
✅ L'adaptateur gère automatiquement ce cas
```

### Dépendances Système
- **Windows** : `secret_envs.dll` 
- **Linux** : `libsecret_envs.so`
- **macOS** : `libsecret_envs.dylib` (ARM) ou `libsecret_envs_intel_macos.dylib` (Intel)

## 🎛️ Personnalisation

### Ajouter un Nouvel Environnement Secret
```python
# Dans la cellule 5, ajoutez :
env_classes = {
    'SecretEnv0': SecretEnv0,
    'SecretEnv1': SecretEnv1, 
    'SecretEnv2': SecretEnv2,
    'SecretEnv3': SecretEnv3,
    'SecretEnv4': SecretEnv4  # <- Nouveau
}
```

### Modifier les Hyperparamètres
```python
# Algorithmes plus exploratoires
SecretOnPolicyMC(adapter, epsilon=0.5, gamma=0.95)

# Plus d'épisodes pour convergence
run_complete_analysis(num_episodes=1500)
```

## 📚 Références Théoriques

Les algorithmes implémentés suivent les spécifications de :
- Sutton & Barto - *Reinforcement Learning: An Introduction*
- Chapitres 5.1-5.7 sur les méthodes Monte Carlo
- Importance Sampling et Off-Policy Learning

## 🏁 Temps d'Exécution Estimé

| Configuration | Temps Estimé | Utilisation |
|---------------|-------------|-------------|
| 200 épisodes  | 2-5 minutes | Tests rapides |
| 800 épisodes  | 5-15 minutes | Analyse complète |
| 1500 épisodes | 15-30 minutes | Recherche approfondie |

---

**🎉 Explorez les mystères des environnements secrets avec Monte Carlo !** 