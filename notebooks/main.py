import streamlit as st
import importlib
import sys
import os
from typing import Any, Dict
import json
from datetime import datetime
import numpy as np

# Ajout des chemins pour l'import dynamique des modules src et game
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../game')))

# Import des environnements RL personnalisés
from game import environments

# Ajout des environnements secrets
from game.secret_env.secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3

# Dictionnaire des environnements disponibles
env_classes = {
    'LineWorld': environments.LineWorld,
    'GridWorld': environments.GridWorld,
    'MontyHallParadox1': environments.MontyHallParadox1,
    'MontyHallParadox2': environments.MontyHallParadox2,
    'RockPaperScissors': environments.RockPaperScissors,
    'SecretEnv0': SecretEnv0,
    'SecretEnv1': SecretEnv1,
    'SecretEnv2': SecretEnv2,
    'SecretEnv3': SecretEnv3,
}

# Dictionnaire des modèles RL disponibles (nom: (module, classe))
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

# Fonction utilitaire pour charger dynamiquement une classe de modèle
# à partir de son nom de module et de classe

def get_model_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

# Fonction utilitaire pour rendre un objet JSON-serializable (conversion numpy → python natif)
def to_serializable(obj):
    import numpy as np
    if isinstance(obj, dict):
        # Convertit les clés tuple en string
        return {str(k): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif hasattr(obj, 'tolist'):  # Pour les numpy arrays
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj

# Fichier où l'historique des tests est sauvegardé
HISTO_FILE = os.path.join(os.path.dirname(__file__), 'test_history.json')

# Fonction pour sauvegarder un résultat de test dans l'historique
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

# Fonction pour charger l'historique des tests
def load_test_history():
    try:
        if os.path.exists(HISTO_FILE):
            with open(HISTO_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []

# Configuration de la page Streamlit (titre, layout)
st.set_page_config(page_title="RLProject", layout="wide")

# Création de deux onglets : Exécution et Historique
tabs = st.tabs(["Exécution", "Historique des tests"])

# Onglet principal : Exécution d'un test RL
with tabs[0]:
    st.title('RLProject - Démonstrateur RL')

    # Sélection de l'environnement et du modèle
    env_name = st.selectbox('Choisissez un environnement RL', list(env_classes.keys()))
    model_name = st.selectbox('Choisissez un modèle RL', list(model_classes.keys()))

    # Instanciation de l'environnement choisi
    env = env_classes[env_name]()
    # Patch sécurité : s'assurer que observation_space et action_space existent
    if not hasattr(env, 'observation_space'):
        if hasattr(env, 'num_states'):
            nS = env.num_states() if callable(env.num_states) else env.num_states
            env.observation_space = type('obj', (), {'n': nS})()
        elif hasattr(env, 'get_state_space'):
            nS = len(env.get_state_space())
            env.observation_space = type('obj', (), {'n': nS})()
    if not hasattr(env, 'action_space'):
        if hasattr(env, 'num_actions'):
            nA = env.num_actions() if callable(env.num_actions) else env.num_actions
            env.action_space = type('obj', (), {'n': nA})()
        elif hasattr(env, 'get_action_space'):
            nA = len(env.get_action_space())
            env.action_space = type('obj', (), {'n': nA})()

    # Affichage dynamique des hyperparamètres selon le modèle choisi
    hyperparams: Dict[str, Any] = {}

    # Affichage des sliders/inputs selon le modèle sélectionné
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
                elif model_name == 'MonteCarloES':
                    model = ModelClass(env, gamma=hyperparams['gamma'])
                    result = model.train(num_episodes=hyperparams['num_episodes'])
                elif model_name in ['OnPolicyMC', 'OffPolicyMC']:
                    model = ModelClass(env, gamma=hyperparams['gamma'], epsilon=hyperparams['epsilon'])
                    result = model.train(num_episodes=hyperparams['num_episodes'])
                elif model_name in ['DynaQ', 'DynaQPlus']:
                    model = ModelClass(env, gamma=hyperparams['gamma'], alpha=hyperparams['alpha'], epsilon=hyperparams['epsilon'])
                    result = model.train(num_episodes=hyperparams['num_episodes'])
                else:
                    model = ModelClass(env)
                    result = model.train()
            except Exception as e:
                import traceback
                st.error(f"Erreur lors de l'entraînement : {e}\n{traceback.format_exc()}")
                st.stop()
            except BaseException as e:
                st.error(f"Erreur inattendue (non-Exception) : {e} (type: {type(e)})")
                st.stop()
            
            # Affichage des résultats d'entraînement sans la clé 'history'
            st.success('Entraînement terminé !')
            train_display = dict(result) if result else {}
            train_display.pop('history', None)
            st.write("**Résultats entraînement :**", train_display)
            
            # Affichage du score final si disponible
            score = None
            final_reward = None
            eval_metrics = {}
            if hasattr(model, 'evaluate'):
                st.subheader('Évaluation du modèle')
                try:
                    eval_result = model.evaluate(num_episodes=100)
                    # Affichage des résultats d'évaluation sans la clé 'history'
                    eval_display = dict(eval_result) if eval_result else {}
                    eval_display.pop('history', None)
                    st.write("**Résultats évaluation :**", eval_display)
                    st.write(eval_result)
                    # Recherche d'un score global dans le résultat d'évaluation
                    if isinstance(eval_result, dict):
                        score = eval_result.get('score')
                        if score is None:
                            for k in ['mean_reward', 'average_reward', 'reward', 'total_reward', 'avg_reward']:
                                if k in eval_result:
                                    score = eval_result[k]
                                    break
                        # Recherche du reward final explicite
                        for k in ['total_reward', 'reward', 'final_reward']:
                            if k in eval_result:
                                final_reward = eval_result[k]
                                break
                        # Affichage visuel agréable des métriques avancées
                        cols = st.columns(3)
                        with cols[0]:
                            if 'avg_reward' in eval_result:
                                st.metric('🎯 Récompense moyenne', f"{eval_result['avg_reward']:.2f}")
                            if 'min_reward' in eval_result:
                                st.metric('⬇️ Récompense min.', f"{eval_result['min_reward']:.2f}")
                            if 'max_reward' in eval_result:
                                st.metric('⬆️ Récompense max.', f"{eval_result['max_reward']:.2f}")
                        with cols[1]:
                            if 'iterations' in eval_result:
                                st.metric('🔁 Itérations/épisodes', eval_result['iterations'])
                            if 'converged' in eval_result:
                                st.metric('✅ Convergence', str(eval_result['converged']))
                        with cols[2]:
                            if 'execution_time' in eval_result:
                                st.metric('⏱️ Temps total (s)', f"{eval_result['execution_time']:.2f}")
                            if score is not None:
                                st.metric('🏆 Score global', score)
                            if final_reward is not None:
                                st.metric('⭐ Reward final', final_reward)
                        # Courbe d'apprentissage
                        if 'learning_curve' in eval_result and eval_result['learning_curve']:
                            st.markdown('**Courbe d’apprentissage (valeur moyenne par itération)**')
                            st.line_chart(eval_result['learning_curve'], height=250, use_container_width=True)
                        eval_metrics = {
                            'avg_reward': eval_result.get('avg_reward'),
                            'min_reward': eval_result.get('min_reward'),
                            'max_reward': eval_result.get('max_reward'),
                            'iterations': eval_result.get('iterations'),
                            'execution_time': eval_result.get('execution_time'),
                            'learning_curve': eval_result.get('learning_curve'),
                        }
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
            # Sauvegarde du test dans l'historique
            save_test_result({
                'datetime': datetime.now().isoformat(),
                'environnement': env_name,
                'modele': model_name,
                'hyperparametres': hyperparams,
                'resultats_train': result,
                'resultats_eval': eval_result if 'eval_result' in locals() else None,
                'score': score,
                'reward_final': final_reward,
                'eval_metrics': eval_metrics
            })

    st.caption('Projet RL ESGI 2025 - Interface Streamlit')

# Onglet secondaire : Historique des tests sauvegardés
with tabs[1]:
    st.title('Historique des tests')
    history = load_test_history()
    if not history:
        st.info("Aucun test sauvegardé pour le moment.")
    else:
        for test in reversed(history):
            with st.expander(f"{test['datetime']} | {test['environnement']} | {test['modele']}"):
                # Affichage des hyperparamètres
                st.write("**Hyperparamètres :**", test['hyperparametres'])
                # Affichage des résultats d'entraînement sans la clé 'history'
                train_display = dict(test['resultats_train']) if test['resultats_train'] else {}
                train_display.pop('history', None)
                st.write("**Résultats entraînement :**", train_display)
                # Affichage des résultats d'évaluation sans la clé 'history'
                eval_display = dict(test['resultats_eval']) if test['resultats_eval'] else {}
                eval_display.pop('history', None)
                st.write("**Résultats évaluation :**", eval_display)
                st.write("**Score :**", test['score'])
                st.write("**Reward final :**", test.get('reward_final', None))
                if 'eval_metrics' in test:
                    st.markdown('---')
                    st.markdown('**Résumé des métriques d’évaluation :**')
                    cols = st.columns(3)
                    with cols[0]:
                        if test['eval_metrics'].get('avg_reward') is not None:
                            st.metric('🎯 Récompense moyenne', f"{test['eval_metrics']['avg_reward']:.2f}")
                        if test['eval_metrics'].get('min_reward') is not None:
                            st.metric('⬇️ Récompense min.', f"{test['eval_metrics']['min_reward']:.2f}")
                        if test['eval_metrics'].get('max_reward') is not None:
                            st.metric('⬆️ Récompense max.', f"{test['eval_metrics']['max_reward']:.2f}")
                    with cols[1]:
                        if test['eval_metrics'].get('iterations') is not None:
                            st.metric('🔁 Itérations/épisodes', test['eval_metrics']['iterations'])
                    with cols[2]:
                        if test['eval_metrics'].get('execution_time') is not None:
                            st.metric('⏱️ Temps total (s)', f"{test['eval_metrics']['execution_time']:.2f}")
                    if test['eval_metrics'].get('learning_curve'):
                        st.markdown('**Courbe d’apprentissage (valeur moyenne par itération)**')
                        st.line_chart(test['eval_metrics']['learning_curve'], height=250, use_container_width=True) 