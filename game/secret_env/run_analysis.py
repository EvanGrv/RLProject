#!/usr/bin/env python3
"""
🚀 LANCEUR D'ANALYSE DES ENVIRONNEMENTS SECRETS

Script simple pour lancer l'analyse complète des environnements secrets.
"""

import os
import sys

def navigate_to_correct_directory():
    """Navigation automatique vers le bon répertoire"""
    print("📁 Recherche du répertoire des environnements secrets...")
    
    # Chemins possibles relatifs au répertoire courant
    possible_paths = [
        '.',  # Déjà dans le bon répertoire
        'game/secret_env',  # Depuis la racine du projet
        '../game/secret_env',  # Depuis un sous-répertoire
        '../../game/secret_env',  # Depuis un sous-sous-répertoire
        './game/secret_env',  # Variante explicite
    ]
    
    for path in possible_paths:
        target_file = os.path.join(path, 'secret_envs_wrapper.py')
        if os.path.exists(target_file):
            abs_path = os.path.abspath(path)
            os.chdir(abs_path)
            print(f"✅ Navigation vers: {abs_path}")
            return True
    
    # Si on ne trouve pas, essayer de détecter depuis le chemin absolu
    current_path = os.getcwd()
    if 'RLProject' in current_path:
        # Construire le chemin vers game/secret_env
        rlproject_index = current_path.find('RLProject')
        if rlproject_index != -1:
            rlproject_path = current_path[:rlproject_index + len('RLProject')]
            secret_env_path = os.path.join(rlproject_path, 'game', 'secret_env')
            
            if os.path.exists(os.path.join(secret_env_path, 'secret_envs_wrapper.py')):
                os.chdir(secret_env_path)
                print(f"✅ Navigation vers: {secret_env_path}")
                return True
    
    print("❌ Impossible de localiser le répertoire des environnements secrets")
    print("💡 Assurez-vous d'être dans ou proche du projet RLProject")
    return False

def check_prerequisites():
    """Vérifie les prérequis avant de lancer l'analyse"""
    print("🔍 Vérification des prérequis...")
    
    # Vérifier le répertoire
    if not os.path.exists('secret_envs_wrapper.py'):
        print("❌ secret_envs_wrapper.py non trouvé dans le répertoire courant")
        return False
    
    # Vérifier le répertoire libs
    if not os.path.exists('libs'):
        print("❌ Erreur: répertoire libs/ non trouvé")
        print("💡 Vérifiez que vous avez tous les fichiers du projet")
        return False
    
    # Vérifier les modules Python requis
    required_modules = ['numpy', 'pandas', 'matplotlib', 'tqdm']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Modules Python manquants: {', '.join(missing_modules)}")
        print("💡 Installez-les avec:")
        print(f"   pip install {' '.join(missing_modules)}")
        return False
    
    print("✅ Tous les prérequis sont satisfaits")
    return True

def test_environments():
    """Test rapide des environnements avant l'analyse complète"""
    print("🎮 Test rapide des environnements...")
    
    try:
        from secret_envs_wrapper import SecretEnv0
        env = SecretEnv0()
        num_states = env.num_states()
        num_actions = env.num_actions()
        print(f"✅ Test réussi - SecretEnv0: {num_states} états, {num_actions} actions")
        return True
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        print("💡 Lancez le diagnostic pour plus de détails:")
        print("   python diagnostic.py")
        return False

def main():
    """Lance l'analyse des environnements secrets"""
    print("🚀 LANCEUR D'ANALYSE DES ENVIRONNEMENTS SECRETS")
    print("=" * 60)
    
    # Navigation automatique vers le bon répertoire
    if not navigate_to_correct_directory():
        print("\n❌ Impossible de localiser les fichiers nécessaires")
        print("💡 Solutions possibles:")
        print("   1. Naviguez manuellement: cd game/secret_env")
        print("   2. Lancez depuis le répertoire RLProject")
        print("   3. Vérifiez que le projet est complet")
        return
    
    # Vérifications préalables
    if not check_prerequisites():
        print("\n❌ Les prérequis ne sont pas satisfaits")
        print("🔧 Lancez d'abord le diagnostic pour identifier les problèmes:")
        print("   python diagnostic.py")
        return
    
    if not test_environments():
        print("\n❌ Les environnements ne fonctionnent pas correctement")
        return
    
    print("\n📊 Démarrage de l'analyse complète...")
    
    # Import et lancement de l'analyseur
    try:
        from environment_stats_analyzer import EnvironmentStatsAnalyzer
        
        analyzer = EnvironmentStatsAnalyzer()
        
        print("🔍 Analyse des environnements...")
        analyzer.analyze_all_environments()
        
        print("📋 Génération du rapport de comparaison...")
        analyzer.generate_comparison_report()
        
        print("📈 Création des visualisations...")
        analyzer.create_visualizations()
        
        print("\n" + "=" * 60)
        print("✅ ANALYSE TERMINÉE AVEC SUCCÈS!")
        print("📁 Fichiers générés dans le répertoire courant:")
        print(f"   📍 {os.getcwd()}")
        print("   - environment_comparison_report.csv")
        print("   - environment_stats_visualization.png")
        print("\n💡 Consultez les résultats pour optimiser vos algorithmes de Deep Learning!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("💡 Vérifiez que environment_stats_analyzer.py est présent")
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        print("🔧 Lancez le diagnostic pour plus d'informations:")
        print("   python diagnostic.py")

if __name__ == "__main__":
    main() 