# Guide d'utilisation - Notebook d'analyse RL

## 🚀 Démarrage rapide

### 1. Ouvrir le notebook
```bash
# Depuis le répertoire du projet
jupyter notebook notebooks/analysis.ipynb
```

### 2. Exécuter l'analyse
- **Option 1 (Recommandée)** : `Kernel > Restart & Run All`
- **Option 2** : Exécuter chaque cellule une par une avec `Shift + Enter`

### 3. Attendre les résultats
- L'analyse peut prendre 2-5 minutes selon votre machine
- Les résultats s'affichent automatiquement

## 📊 Ce que vous obtenez

### Tableau comparatif
- Performances de tous les algorithmes sur tous les environnements
- Métriques : récompense, temps d'entraînement, taux de succès, etc.

### Recommandations
- Meilleur algorithme par environnement
- Algorithme le plus rapide
- Algorithme le plus efficace

### Fichier CSV
- `results_comparison.csv` avec toutes les données
- Peut être ouvert dans Excel/LibreOffice

## 🔧 Résolution des problèmes

### Erreur "No module named 'numpy'"
**Solution** : Exécutez la première cellule qui installe automatiquement les dépendances

### Erreur d'import des algorithmes
**Solution** : Vérifiez que vous êtes dans le bon répertoire :
```bash
# Vous devriez être dans le dossier du projet
ls  # Doit afficher : game/, src/, notebooks/, etc.
```

### Erreur de mémoire
**Solution** : Réduisez le nombre d'épisodes dans la cellule 4 :
```python
# Changez num_episodes de 500 à 100
'MonteCarloES': {'gamma': 0.99, 'num_episodes': 100},
```

## 📈 Interprétation des résultats

### Métriques expliquées

- **Récompense moyenne** : Plus élevée = meilleur
- **Taux de succès** : Pourcentage d'épisodes réussis
- **Temps d'entraînement** : Plus bas = plus rapide
- **Étapes moyennes** : Nombre moyen d'actions par épisode

### Algorithmes par type

- **DP** (Dynamic Programming) : Rapides mais nécessitent un modèle
- **MC** (Monte Carlo) : Bons pour les épisodes complets
- **TD** (Temporal Difference) : Équilibre entre vitesse et performance
- **Planning** : Combinent apprentissage et planification

## 🎯 Conseils d'utilisation

### Pour des tests rapides
Réduisez le nombre d'épisodes dans la configuration :
```python
num_episodes': 100  # Au lieu de 500
```

### Pour une analyse détaillée
Gardez les paramètres par défaut et laissez tourner l'analyse complète.

### Pour des environnements spécifiques
Commentez les environnements non désirés dans la cellule 4 :
```python
environments = {
    'LineWorld': LineWorld(length=8),
    # 'GridWorld': GridWorld(n_rows=5, n_cols=5),  # Commenté
    # 'MontyHall1': MontyHallParadox1(),           # Commenté
    # 'MontyHall2': MontyHallParadox2()            # Commenté
}
```

---

**🎯 Bonne analyse !** 