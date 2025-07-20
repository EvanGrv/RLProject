#!/usr/bin/env python3
"""
🕵️ ANALYSEUR DE STATISTIQUES POUR ENVIRONNEMENTS SECRETS

Ce script analyse les environnements secrets et extrait des statistiques 
détaillées pour l'entraînement d'algorithmes de deep learning.

Author: Environment Statistics Analyzer
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class EnvironmentStatsAnalyzer:
    """Analyseur de statistiques pour les environnements secrets"""
    
    def __init__(self):
        # Vérification et changement de répertoire si nécessaire
        self._ensure_correct_directory()
        
        # Import dynamique des environnements après vérification du répertoire
        try:
            from secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3
            self.env_classes = {
                'SecretEnv0': SecretEnv0,
                'SecretEnv1': SecretEnv1,
                'SecretEnv2': SecretEnv2,
                'SecretEnv3': SecretEnv3
            }
            print("✅ Environnements secrets chargés avec succès")
        except Exception as e:
            print(f"❌ Erreur lors du chargement des environnements: {e}")
            print("💡 Vérifiez que vous êtes dans le bon répertoire et que les DLL sont présentes")
            self.env_classes = {}
            
        self.stats = {}
        self.simulation_results = {}
    
    def _ensure_correct_directory(self):
        """S'assurer qu'on est dans le bon répertoire pour charger les DLL"""
        current_dir = os.getcwd()
        
        # Vérifier si on est dans le bon répertoire
        if not os.path.exists('secret_envs_wrapper.py'):
            # Essayer de naviguer vers le bon répertoire
            potential_paths = [
                'game/secret_env',
                '../game/secret_env',
                '../../game/secret_env'
            ]
            
            for path in potential_paths:
                if os.path.exists(os.path.join(path, 'secret_envs_wrapper.py')):
                    os.chdir(path)
                    print(f"📁 Changement de répertoire vers: {os.getcwd()}")
                    break
            else:
                print("⚠️ Impossible de trouver le répertoire des environnements secrets")
                return False
        
        # Vérifier que les DLL existent
        libs_dir = 'libs'
        if not os.path.exists(libs_dir):
            print(f"❌ Répertoire {libs_dir} non trouvé dans {os.getcwd()}")
            return False
            
        # Vérifier les DLL spécifiques selon l'OS
        import platform
        if platform.system().lower() == "windows":
            dll_path = os.path.join(libs_dir, "secret_envs.dll")
        elif platform.system().lower() == "linux":
            dll_path = os.path.join(libs_dir, "libsecret_envs.so")
        elif platform.system().lower() == "darwin":
            if "intel" in platform.processor().lower():
                dll_path = os.path.join(libs_dir, "libsecret_envs_intel_macos.dylib")
            else:
                dll_path = os.path.join(libs_dir, "libsecret_envs.dylib")
        
        if not os.path.exists(dll_path):
            print(f"❌ Bibliothèque native non trouvée: {dll_path}")
            print(f"📁 Contenu du répertoire libs: {os.listdir(libs_dir) if os.path.exists(libs_dir) else 'N/A'}")
            return False
            
        print(f"✅ Bibliothèque native trouvée: {dll_path}")
        return True
        
    def analyze_basic_properties(self, env_name, env_class):
        """Analyse les propriétés de base de l'environnement"""
        print(f"🔍 Analyse de {env_name}...")
        
        try:
            env = env_class()
            
            basic_stats = {
                'num_states': env.num_states(),
                'num_actions': env.num_actions(),
                'num_rewards': env.num_rewards(),
                'rewards': [env.reward(i) for i in range(env.num_rewards())],
                'state_space_size': env.num_states(),
                'action_space_size': env.num_actions()
            }
            
            # Calcul de la complexité
            basic_stats['state_action_complexity'] = basic_stats['num_states'] * basic_stats['num_actions']
            basic_stats['estimated_complexity_class'] = self._classify_complexity(basic_stats['state_action_complexity'])
            
            return basic_stats
            
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse de {env_name}: {e}")
            return None
    
    def _classify_complexity(self, state_action_complexity):
        """Classifie la complexité de l'environnement"""
        if state_action_complexity < 100:
            return "TRES_SIMPLE"
        elif state_action_complexity < 1000:
            return "SIMPLE"
        elif state_action_complexity < 10000:
            return "MOYEN"
        elif state_action_complexity < 100000:
            return "COMPLEXE"
        else:
            return "TRES_COMPLEXE"
    
    def analyze_transition_structure(self, env_name, env_class, sample_size=100):
        """Analyse la structure des transitions"""
        print(f"🔗 Analyse des transitions pour {env_name}...")
        
        try:
            env = env_class()
            num_states = env.num_states()
            num_actions = env.num_actions()
            num_rewards = env.num_rewards()
            
            # Échantillonnage pour éviter une analyse exhaustive sur de gros environnements
            sample_states = min(sample_size, num_states)
            sample_actions = min(sample_size, num_actions)
            
            transitions_stats = {
                'sparsity': 0,  # Proportion de transitions nulles
                'deterministic_transitions': 0,  # Nombre de transitions déterministes
                'average_transition_prob': 0,
                'max_transition_prob': 0,
                'connectivity': 0  # Mesure de connectivité du graphe d'états
            }
            
            total_transitions = 0
            zero_transitions = 0
            deterministic_count = 0
            all_probs = []
            
            state_connections = defaultdict(set)
            
            for s in range(0, num_states, max(1, num_states // sample_states)):
                for a in range(0, num_actions, max(1, num_actions // sample_actions)):
                    state_probs = []
                    for s_p in range(num_states):
                        for r_idx in range(num_rewards):
                            prob = env.p(s, a, s_p, r_idx)
                            total_transitions += 1
                            
                            if prob == 0:
                                zero_transitions += 1
                            else:
                                all_probs.append(prob)
                                state_connections[s].add(s_p)
                                
                            if prob == 1.0:
                                deterministic_count += 1
            
            if total_transitions > 0:
                transitions_stats['sparsity'] = zero_transitions / total_transitions
                transitions_stats['deterministic_transitions'] = deterministic_count
                
            if all_probs:
                transitions_stats['average_transition_prob'] = np.mean(all_probs)
                transitions_stats['max_transition_prob'] = np.max(all_probs)
                
            # Connectivité approximative
            transitions_stats['connectivity'] = len(state_connections) / num_states if num_states > 0 else 0
            
            return transitions_stats
            
        except Exception as e:
            print(f"❌ Erreur analyse transitions {env_name}: {e}")
            return {}
    
    def simulate_episodes(self, env_name, env_class, num_episodes=50, max_steps=1000):
        """Simule des épisodes pour analyser la dynamique"""
        print(f"🎮 Simulation de {num_episodes} épisodes pour {env_name}...")
        
        try:
            episode_lengths = []
            episode_scores = []
            action_distributions = []
            state_visits = defaultdict(int)
            
            for episode in tqdm(range(num_episodes), desc=f"Episodes {env_name}"):
                env = env_class()
                env.reset()
                
                episode_length = 0
                episode_actions = []
                
                for step in range(max_steps):
                    if env.is_game_over():
                        break
                        
                    current_state = env.state_id()
                    state_visits[current_state] += 1
                    
                    available_actions = env.available_actions()
                    if len(available_actions) == 0:
                        break
                        
                    # Action aléatoire pour exploration
                    action = np.random.choice(available_actions)
                    episode_actions.append(action)
                    
                    env.step(action)
                    episode_length += 1
                
                episode_lengths.append(episode_length)
                episode_scores.append(env.score())
                action_distributions.extend(episode_actions)
            
            simulation_stats = {
                'avg_episode_length': np.mean(episode_lengths),
                'std_episode_length': np.std(episode_lengths),
                'min_episode_length': np.min(episode_lengths),
                'max_episode_length': np.max(episode_lengths),
                'avg_score': np.mean(episode_scores),
                'std_score': np.std(episode_scores),
                'min_score': np.min(episode_scores),
                'max_score': np.max(episode_scores),
                'action_distribution': dict(Counter(action_distributions)),
                'state_visit_distribution': dict(state_visits),
                'convergence_difficulty': self._estimate_convergence_difficulty(episode_lengths, episode_scores)
            }
            
            return simulation_stats
            
        except Exception as e:
            print(f"❌ Erreur simulation {env_name}: {e}")
            return {}
    
    def _estimate_convergence_difficulty(self, episode_lengths, episode_scores):
        """Estime la difficulté de convergence basée sur la variabilité"""
        length_cv = np.std(episode_lengths) / np.mean(episode_lengths) if np.mean(episode_lengths) > 0 else 0
        score_cv = np.std(episode_scores) / np.mean(episode_scores) if np.mean(episode_scores) > 0 else 0
        
        avg_cv = (length_cv + score_cv) / 2
        
        if avg_cv < 0.2:
            return "FACILE"
        elif avg_cv < 0.5:
            return "MOYEN"
        elif avg_cv < 1.0:
            return "DIFFICILE"
        else:
            return "TRES_DIFFICILE"
    
    def estimate_training_requirements(self, env_name, basic_stats, simulation_stats):
        """Estime les exigences d'entraînement pour deep learning"""
        
        # Facteurs de complexité
        state_complexity = basic_stats.get('num_states', 1)
        action_complexity = basic_stats.get('num_actions', 1)
        avg_episode_length = simulation_stats.get('avg_episode_length', 100)
        convergence_difficulty = simulation_stats.get('convergence_difficulty', 'MOYEN')
        
        # Estimation du nombre d'épisodes nécessaires
        base_episodes = state_complexity * action_complexity * 0.1
        
        # Facteurs multiplicateurs
        length_factor = min(avg_episode_length / 50, 5)  # Cap à 5x
        
        difficulty_multipliers = {
            'FACILE': 1.0,
            'MOYEN': 2.0,
            'DIFFICILE': 4.0,
            'TRES_DIFFICILE': 8.0
        }
        
        difficulty_factor = difficulty_multipliers.get(convergence_difficulty, 2.0)
        
        estimated_episodes = int(base_episodes * length_factor * difficulty_factor)
        
        # Estimation du temps d'entraînement (en supposant 10ms par step)
        estimated_steps_per_episode = avg_episode_length
        total_steps = estimated_episodes * estimated_steps_per_episode
        estimated_time_minutes = (total_steps * 0.01) / 60  # 10ms par step
        
        # Recommandations d'architecture
        hidden_size_recommendation = min(max(state_complexity * 2, 64), 512)
        
        training_requirements = {
            'estimated_episodes_needed': estimated_episodes,
            'estimated_total_steps': int(total_steps),
            'estimated_training_time_minutes': estimated_time_minutes,
            'recommended_batch_size': min(max(estimated_episodes // 100, 32), 256),
            'recommended_hidden_size': hidden_size_recommendation,
            'recommended_learning_rate': 0.001 if convergence_difficulty in ['FACILE', 'MOYEN'] else 0.0001,
            'recommended_exploration_rate': 0.1 if convergence_difficulty == 'FACILE' else 0.3,
            'memory_requirements_mb': (state_complexity * action_complexity * 4) / 1024 / 1024  # Approximation
        }
        
        return training_requirements
    
    def analyze_all_environments(self):
        """Lance l'analyse complète de tous les environnements"""
        if not self.env_classes:
            print("❌ Aucun environnement disponible pour l'analyse")
            print("💡 Vérifiez que vous êtes dans le bon répertoire et que les DLL sont présentes")
            return
            
        print("🚀 ANALYSE COMPLÈTE DES ENVIRONNEMENTS SECRETS")
        print("=" * 60)
        
        successful_analyses = 0
        
        for env_name, env_class in self.env_classes.items():
            print(f"\n📊 ANALYSE DE {env_name}")
            print("-" * 40)
            
            try:
                # Analyse des propriétés de base
                basic_stats = self.analyze_basic_properties(env_name, env_class)
                if not basic_stats:
                    print(f"⚠️ Échec de l'analyse de base pour {env_name}")
                    continue
                    
                # Analyse des transitions
                transition_stats = self.analyze_transition_structure(env_name, env_class)
                
                # Simulation d'épisodes
                simulation_stats = self.simulate_episodes(env_name, env_class)
                
                # Estimation des exigences d'entraînement
                training_requirements = self.estimate_training_requirements(
                    env_name, basic_stats, simulation_stats
                )
                
                # Stockage des résultats
                self.stats[env_name] = {
                    'basic': basic_stats,
                    'transitions': transition_stats,
                    'simulation': simulation_stats,
                    'training': training_requirements
                }
                
                # Affichage des résultats
                self._print_environment_summary(env_name)
                successful_analyses += 1
                
            except Exception as e:
                print(f"❌ Erreur complète lors de l'analyse de {env_name}: {e}")
                continue
        
        print(f"\n✅ Analyse terminée: {successful_analyses}/{len(self.env_classes)} environnements analysés")
    
    def _print_environment_summary(self, env_name):
        """Affiche un résumé des statistiques pour un environnement"""
        stats = self.stats[env_name]
        basic = stats['basic']
        simulation = stats['simulation']
        training = stats['training']
        
        print(f"\n📈 RÉSUMÉ {env_name}:")
        print(f"   🏗️  États: {basic['num_states']} | Actions: {basic['num_actions']}")
        print(f"   🎯  Complexité: {basic['estimated_complexity_class']}")
        print(f"   ⏱️  Longueur épisode moy: {simulation.get('avg_episode_length', 'N/A'):.1f}")
        print(f"   🏆  Score moyen: {simulation.get('avg_score', 'N/A'):.2f}")
        print(f"   🧠  Difficulté convergence: {simulation.get('convergence_difficulty', 'N/A')}")
        print(f"   🎓  Épisodes recommandés: {training['estimated_episodes_needed']:,}")
        print(f"   ⏰  Temps d'entraînement estimé: {training['estimated_training_time_minutes']:.1f} min")
        print(f"   🔧  Taille cachée recommandée: {training['recommended_hidden_size']}")
        print(f"   📚  Learning rate recommandé: {training['recommended_learning_rate']}")
    
    def generate_comparison_report(self):
        """Génère un rapport de comparaison entre environnements"""
        if not self.stats:
            print("❌ Aucune statistique disponible pour la comparaison")
            return
            
        print("\n" + "=" * 60)
        print("📊 RAPPORT DE COMPARAISON DES ENVIRONNEMENTS")
        print("=" * 60)
        
        # Tableau comparatif
        comparison_data = []
        for env_name, stats in self.stats.items():
            basic = stats['basic']
            simulation = stats['simulation']
            training = stats['training']
            
            comparison_data.append({
                'Environnement': env_name,
                'États': basic['num_states'],
                'Actions': basic['num_actions'],
                'Complexité': basic['estimated_complexity_class'],
                'Longueur Épisode': f"{simulation.get('avg_episode_length', 0):.1f}",
                'Score Moyen': f"{simulation.get('avg_score', 0):.2f}",
                'Difficulté': simulation.get('convergence_difficulty', 'N/A'),
                'Épisodes Requis': f"{training['estimated_episodes_needed']:,}",
                'Temps (min)': f"{training['estimated_training_time_minutes']:.1f}",
                'Hidden Size': training['recommended_hidden_size'],
                'Learning Rate': training['recommended_learning_rate']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\n📋 TABLEAU COMPARATIF:")
        print(df_comparison.to_string(index=False))
        
        # Recommandations globales
        print(f"\n🎯 RECOMMANDATIONS GLOBALES:")
        print("-" * 30)
        
        easiest_env = min(self.stats.items(), 
                         key=lambda x: x[1]['training']['estimated_episodes_needed'])
        hardest_env = max(self.stats.items(), 
                         key=lambda x: x[1]['training']['estimated_episodes_needed'])
        
        print(f"🟢 Plus facile à entraîner: {easiest_env[0]}")
        print(f"🔴 Plus difficile à entraîner: {hardest_env[0]}")
        
        avg_episodes = np.mean([stats['training']['estimated_episodes_needed'] 
                               for stats in self.stats.values()])
        print(f"📊 Nombre moyen d'épisodes requis: {avg_episodes:,.0f}")
        
        # Sauvegarde en CSV
        df_comparison.to_csv('environment_comparison_report.csv', index=False)
        print(f"\n💾 Rapport sauvegardé: environment_comparison_report.csv")
    
    def create_visualizations(self):
        """Crée des visualisations des statistiques"""
        if not self.stats:
            print("❌ Aucune statistique disponible pour les visualisations")
            return
            
        try:
            plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        except:
            pass  # Utiliser le style par défaut si seaborn n'est pas disponible
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🕵️ Analyse des Environnements Secrets', fontsize=16, fontweight='bold')
        
        # 1. Complexité des environnements
        env_names = list(self.stats.keys())
        complexities = [stats['basic']['state_action_complexity'] for stats in self.stats.values()]
        
        axes[0, 0].bar(env_names, complexities, color='skyblue')
        axes[0, 0].set_title('🏗️ Complexité (États × Actions)')
        axes[0, 0].set_ylabel('Complexité')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Temps d'entraînement estimé
        training_times = [stats['training']['estimated_training_time_minutes'] 
                         for stats in self.stats.values()]
        
        axes[0, 1].bar(env_names, training_times, color='lightcoral')
        axes[0, 1].set_title('⏰ Temps d\'Entraînement Estimé (min)')
        axes[0, 1].set_ylabel('Minutes')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Longueur moyenne des épisodes
        episode_lengths = [stats['simulation'].get('avg_episode_length', 0) 
                          for stats in self.stats.values()]
        
        axes[1, 0].bar(env_names, episode_lengths, color='lightgreen')
        axes[1, 0].set_title('📏 Longueur Moyenne des Épisodes')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Épisodes d'entraînement requis
        required_episodes = [stats['training']['estimated_episodes_needed'] 
                           for stats in self.stats.values()]
        
        axes[1, 1].bar(env_names, required_episodes, color='gold')
        axes[1, 1].set_title('🎓 Épisodes d\'Entraînement Requis')
        axes[1, 1].set_ylabel('Nombre d\'épisodes')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('environment_stats_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualisations sauvegardées: environment_stats_visualization.png")


def main():
    """Fonction principale"""
    print("🕵️ ANALYSEUR DE STATISTIQUES POUR ENVIRONNEMENTS SECRETS")
    print("=" * 60)
    print("Ce script analyse les environnements secrets pour l'entraînement Deep Learning")
    print()
    
    analyzer = EnvironmentStatsAnalyzer()
    
    try:
        # Analyse complète
        analyzer.analyze_all_environments()
        
        # Rapport de comparaison
        analyzer.generate_comparison_report()
        
        # Visualisations
        analyzer.create_visualizations()
        
        print("\n" + "=" * 60)
        print("✅ ANALYSE TERMINÉE AVEC SUCCÈS!")
        print("📁 Fichiers générés:")
        print("   - environment_comparison_report.csv")
        print("   - environment_stats_visualization.png")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 