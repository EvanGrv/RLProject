import streamlit as st
import importlib
import sys
import os
from typing import Any, Dict
import json
from datetime import datetime
import numpy as np

# Ajout des chemins pour l'import dynamique
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../game')))

# Import dynamique des environnements
from game import environments

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

def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj

# Fichier pour sauvegarder l'historique des tests
HISTO_FILE = os.path.join(os.path.dirname(__file__), 'test_history.json')

def save_test_result(result: dict):
    try:
        if os.path.exists(HISTO_FILE):
            with open(HISTO_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
    except Exception:
        data = []
    data.append(result)
    with open(HISTO_FILE, 'w', encoding='utf-8') as f:
        json.dump(to_serializable(data), f, indent=2, ensure_ascii=False)

def load_test_history():
    try:
        if os.path.exists(HISTO_FILE):
            with open(HISTO_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []

st.set_page_config(page_title="RLProject", layout="wide")

tabs = st.tabs(["Exécution", "Historique des tests"])

with tabs[0]:
    st.title('RLProject - Démonstrateur RL')

    # Sélection de l'environnement
    env_name = st.selectbox('Choisissez un environnement RL', list(env_classes.keys()))

    # Sélection du modèle
    model_name = st.selectbox('Choisissez un modèle RL', list(model_classes.keys()))

    # Instanciation de l'environnement
    env = env_classes[env_name]()

    # Affichage des hyperparamètres dynamiquement selon le modèle
    hyperparams: Dict[str, Any] = {}

    if model_name in ['PolicyIteration', 'ValueIteration']:
        gamma = st.slider('Gamma (facteur d\'actualisation)', 0.0, 1.0, 0.99, 0.01)
        theta = st.number_input('Theta (seuil de convergence)', min_value=1e-8, max_value=1e-2, value=1e-6, step=1e-8, format="%e")
        hyperparams = {'gamma': gamma, 'theta': theta}
    elif model_name in ['Sarsa', 'QLearning', 'ExpectedSarsa', 'DynaQ', 'DynaQPlus']:
        gamma = st.slider('Gamma', 0.0, 1.0, 0.99, 0.01)
        alpha = st.slider('Alpha (taux d\'apprentissage)', 0.0, 1.0, 0.1, 0.01)
        epsilon = st.slider('Epsilon (exploration)', 0.0, 1.0, 0.1, 0.01)
        episodes = st.number_input('Nombre d\'épisodes', min_value=1, max_value=10000, value=500)
        hyperparams = {'gamma': gamma, 'alpha': alpha, 'epsilon': epsilon, 'num_episodes': episodes}
    elif model_name in ['MonteCarloES', 'OnPolicyMC', 'OffPolicyMC']:
        gamma = st.slider('Gamma', 0.0, 1.0, 0.99, 0.01)
        epsilon = st.slider('Epsilon', 0.0, 1.0, 0.1, 0.01)
        episodes = st.number_input('Nombre d\'épisodes', min_value=1, max_value=10000, value=500)
        hyperparams = {'gamma': gamma, 'epsilon': epsilon, 'num_episodes': episodes}
    else:
        st.info("Aucun hyperparamètre spécifique pour ce modèle.")

    # Lancement de l'entraînement
    if st.button('Lancer l\'entraînement'):
        with st.spinner('Entraînement en cours...'):
            module_name, class_name = model_classes[model_name]
            ModelClass = get_model_class(module_name, class_name)
            # Instanciation du modèle avec les bons hyperparamètres
            try:
                if model_name in ['PolicyIteration', 'ValueIteration']:
                    model = ModelClass(env, gamma=hyperparams['gamma'], theta=hyperparams['theta'])
                    result = model.train()
                elif model_name in ['Sarsa', 'QLearning', 'ExpectedSarsa']:
                    model = ModelClass(env, gamma=hyperparams['gamma'], alpha=hyperparams['alpha'], epsilon=hyperparams['epsilon'])
                    result = model.train(num_episodes=hyperparams['num_episodes'])
                elif model_name in ['MonteCarloES', 'OnPolicyMC', 'OffPolicyMC']:
                    model = ModelClass(env, gamma=hyperparams['gamma'], epsilon=hyperparams['epsilon'])
                    result = model.train(num_episodes=hyperparams['num_episodes'])
                elif model_name in ['DynaQ', 'DynaQPlus']:
                    model = ModelClass(env, gamma=hyperparams['gamma'], alpha=hyperparams['alpha'], epsilon=hyperparams['epsilon'])
                    result = model.train(num_episodes=hyperparams['num_episodes'])
                else:
                    model = ModelClass(env)
                    result = model.train()
            except Exception as e:
                st.error(f"Erreur lors de l'entraînement : {e}")
                st.stop()
            
            # Affichage des résultats pas à pas et des métriques
            st.success('Entraînement terminé !')
            if isinstance(result, dict):
                for k, v in result.items():
                    st.write(f"**{k}** : {v}")
            else:
                st.write(result)
            
            # Affichage du score final si disponible
            score = None
            if hasattr(model, 'evaluate'):
                st.subheader('Évaluation du modèle')
                try:
                    eval_result = model.evaluate(num_episodes=100)
                    st.write(eval_result)
                    # Ajout d'un score global si possible
                    if isinstance(eval_result, dict):
                        score = eval_result.get('score')
                        if score is None:
                            for k in ['mean_reward', 'average_reward', 'reward', 'total_reward', 'avg_reward']:
                                if k in eval_result:
                                    score = eval_result[k]
                                    break
                        if score is not None:
                            st.metric('Score global', score)
                except Exception as e:
                    st.warning(f"Erreur lors de l'évaluation : {e}")
            
            # Affichage d'autres métriques pertinentes si disponibles
            if hasattr(model, 'total_reward'):
                st.metric('Récompense totale', getattr(model, 'total_reward'))
            if hasattr(model, 'steps_count'):
                st.metric('Nombre d\'étapes', getattr(model, 'steps_count'))
            if hasattr(model, 'wins'):
                st.metric('Victoires', getattr(model, 'wins'))
            if hasattr(model, 'total_games'):
                st.metric('Nombre de parties', getattr(model, 'total_games'))

            # Sauvegarde du test
            save_test_result({
                'datetime': datetime.now().isoformat(),
                'environnement': env_name,
                'modele': model_name,
                'hyperparametres': hyperparams,
                'resultats_train': result,
                'resultats_eval': eval_result if 'eval_result' in locals() else None,
                'score': score
            })

    st.caption('Projet RL ESGI 2025 - Interface Streamlit')

with tabs[1]:
    st.title('Historique des tests')
    history = load_test_history()
    if not history:
        st.info("Aucun test sauvegardé pour le moment.")
    else:
        for test in reversed(history):
            with st.expander(f"{test['datetime']} | {test['environnement']} | {test['modele']}"):
                st.write("**Hyperparamètres :**", test['hyperparametres'])
                st.write("**Résultats entraînement :**", test['resultats_train'])
                st.write("**Résultats évaluation :**", test['resultats_eval'])
                st.write("**Score :**", test['score']) 