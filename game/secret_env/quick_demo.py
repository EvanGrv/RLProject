#!/usr/bin/env python3
"""
🎯 DÉMONSTRATION RAPIDE DES ENVIRONNEMENTS SECRETS

Script pour une démonstration rapide des capacités des environnements.
"""

import os
import numpy as np

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

def quick_demo():
    """Démonstration rapide des 4 environnements"""
    print("🎯 DÉMONSTRATION RAPIDE DES ENVIRONNEMENTS SECRETS")
    print("=" * 60)
    
    # Navigation automatique vers le bon répertoire
    if not navigate_to_correct_directory():
        print("\n❌ Impossible de localiser les fichiers nécessaires")
        print("💡 Solutions possibles:")
        print("   1. Naviguez manuellement: cd game/secret_env")
        print("   2. Lancez depuis le répertoire RLProject")
        print("   3. Vérifiez que le projet est complet")
        return
    
    # Import des environnements après navigation
    try:
        from secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3
        print("✅ Import des environnements réussi")
    except Exception as e:
        print(f"❌ Erreur d'import des environnements: {e}")
        print("💡 Vérifiez que les DLL sont présentes dans le répertoire libs/")
        return
    
    env_classes = {
        'SecretEnv0': SecretEnv0,
        'SecretEnv1': SecretEnv1, 
        'SecretEnv2': SecretEnv2,
        'SecretEnv3': SecretEnv3
    }
    
    successful_tests = 0
    
    for env_name, env_class in env_classes.items():
        print(f"\n🔍 {env_name}:")
        print("-" * 20)
        
        try:
            env = env_class()
            
            # Statistiques de base
            print(f"   États: {env.num_states()}")
            print(f"   Actions: {env.num_actions()}")
            print(f"   Récompenses: {env.num_rewards()}")
            
            # Récompenses disponibles
            rewards = [env.reward(i) for i in range(env.num_rewards())]
            print(f"   Valeurs récompenses: {rewards}")
            
            # Test d'un épisode court
            env.reset()
            steps = 0
            max_steps = 20
            
            print(f"   État initial: {env.state_id()}")
            print(f"   Actions disponibles: {env.available_actions()}")
            
            # Quelques steps
            while not env.is_game_over() and steps < max_steps:
                available = env.available_actions()
                if len(available) == 0:
                    break
                action = np.random.choice(available)
                env.step(action)
                steps += 1
            
            print(f"   Steps effectués: {steps}")
            print(f"   Score final: {env.score():.2f}")
            print(f"   Game over: {env.is_game_over()}")
            print(f"   ✅ Test réussi!")
            successful_tests += 1
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
    
    # Résumé final
    print(f"\n📊 RÉSUMÉ:")
    print(f"   ✅ {successful_tests}/{len(env_classes)} environnements testés avec succès")
    
    if successful_tests == len(env_classes):
        print(f"   🎉 Tous les environnements fonctionnent!")
        print(f"   🚀 Vous pouvez maintenant lancer l'analyse complète:")
        print(f"      python run_analysis.py")
    elif successful_tests > 0:
        print(f"   ⚠️ Certains environnements ont des problèmes")
        print(f"   🔧 Lancez le diagnostic: python diagnostic.py")
    else:
        print(f"   ❌ Aucun environnement ne fonctionne")
        print(f"   🔧 Lancez le diagnostic: python diagnostic.py")

if __name__ == "__main__":
    quick_demo() 