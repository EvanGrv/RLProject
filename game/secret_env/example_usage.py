#!/usr/bin/env python3
"""
📝 EXEMPLE D'UTILISATION DE L'ANALYSEUR

Exemple simple montrant comment utiliser l'analyseur pour obtenir 
des statistiques spécifiques sur les environnements secrets.
"""

from environment_stats_analyzer import EnvironmentStatsAnalyzer
from secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3

def example_basic_analysis():
    """Exemple d'analyse basique d'un seul environnement"""
    print("📊 EXEMPLE: Analyse d'un seul environnement")
    print("=" * 50)
    
    analyzer = EnvironmentStatsAnalyzer()
    
    # Analyse de SecretEnv0 seulement
    env_name = "SecretEnv0"
    env_class = SecretEnv0
    
    print(f"🔍 Analyse de {env_name}...")
    
    # Propriétés de base
    basic_stats = analyzer.analyze_basic_properties(env_name, env_class)
    print(f"✅ États: {basic_stats['num_states']}, Actions: {basic_stats['num_actions']}")
    print(f"✅ Complexité: {basic_stats['estimated_complexity_class']}")
    
    # Simulation rapide (seulement 10 épisodes pour l'exemple)
    simulation_stats = analyzer.simulate_episodes(env_name, env_class, num_episodes=10)
    print(f"✅ Longueur épisode moyenne: {simulation_stats['avg_episode_length']:.1f}")
    print(f"✅ Score moyen: {simulation_stats['avg_score']:.2f}")
    
    # Estimation d'entraînement
    training_req = analyzer.estimate_training_requirements(env_name, basic_stats, simulation_stats)
    print(f"✅ Épisodes d'entraînement recommandés: {training_req['estimated_episodes_needed']:,}")
    print(f"✅ Learning rate recommandé: {training_req['recommended_learning_rate']}")
    print(f"✅ Taille de couche cachée: {training_req['recommended_hidden_size']}")

def example_comparison():
    """Exemple de comparaison rapide entre environnements"""
    print("\n📊 EXEMPLE: Comparaison rapide")
    print("=" * 50)
    
    analyzer = EnvironmentStatsAnalyzer()
    envs = [("SecretEnv0", SecretEnv0), ("SecretEnv1", SecretEnv1)]
    
    results = {}
    
    for env_name, env_class in envs:
        print(f"🔍 {env_name}...")
        basic = analyzer.analyze_basic_properties(env_name, env_class)
        simulation = analyzer.simulate_episodes(env_name, env_class, num_episodes=5)
        training = analyzer.estimate_training_requirements(env_name, basic, simulation)
        
        results[env_name] = {
            'complexité': basic['estimated_complexity_class'],
            'épisodes_requis': training['estimated_episodes_needed'],
            'temps_minutes': training['estimated_training_time_minutes']
        }
    
    print("\n📋 COMPARAISON:")
    for env_name, stats in results.items():
        print(f"   {env_name}: {stats['complexité']} - {stats['épisodes_requis']:,} épisodes - {stats['temps_minutes']:.1f} min")

def example_get_specific_recommendations():
    """Exemple pour obtenir des recommandations spécifiques pour DQN"""
    print("\n🎯 EXEMPLE: Recommandations pour DQN")
    print("=" * 50)
    
    analyzer = EnvironmentStatsAnalyzer()
    
    # Analyser SecretEnv2 par exemple
    env_name = "SecretEnv2"
    env_class = SecretEnv2
    
    basic = analyzer.analyze_basic_properties(env_name, env_class)
    simulation = analyzer.simulate_episodes(env_name, env_class, num_episodes=8)
    training = analyzer.estimate_training_requirements(env_name, basic, simulation)
    
    print(f"🤖 Configuration DQN recommandée pour {env_name}:")
    print(f"   📐 Architecture: Input({basic['num_states']}) -> Hidden({training['recommended_hidden_size']}) -> Output({basic['num_actions']})")
    print(f"   📚 Learning Rate: {training['recommended_learning_rate']}")
    print(f"   🎯 Episodes: {training['estimated_episodes_needed']:,}")
    print(f"   📦 Batch Size: {training['recommended_batch_size']}")
    print(f"   🔍 Exploration Rate: {training['recommended_exploration_rate']}")
    print(f"   💾 Mémoire requise: {training['memory_requirements_mb']:.1f} MB")

if __name__ == "__main__":
    print("🕵️ EXEMPLES D'UTILISATION DE L'ANALYSEUR")
    print("=" * 60)
    
    try:
        example_basic_analysis()
        example_comparison()
        example_get_specific_recommendations()
        
        print("\n" + "=" * 60)
        print("✅ EXEMPLES TERMINÉS AVEC SUCCÈS!")
        print("💡 Pour une analyse complète, lancez: python run_analysis.py")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("💡 Assurez-vous que les environnements secrets fonctionnent correctement") 