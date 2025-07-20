#!/usr/bin/env python3
"""
🔧 DIAGNOSTIC DES ENVIRONNEMENTS SECRETS

Script de diagnostic pour identifier et résoudre les problèmes 
avec les environnements secrets avant l'analyse.
"""

import os
import sys
import platform

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

def diagnostic_complete():
    """Effectue un diagnostic complet du système"""
    print("🔧 DIAGNOSTIC DES ENVIRONNEMENTS SECRETS")
    print("=" * 50)
    
    # 1. Vérification du système
    print("1. 🖥️ INFORMATIONS SYSTÈME:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.architecture()[0]}")
    print(f"   Processeur: {platform.processor()}")
    print(f"   Python: {sys.version}")
    
    # 2. Vérification du répertoire
    print("\n2. 📁 VÉRIFICATION DU RÉPERTOIRE:")
    current_dir = os.getcwd()
    print(f"   Répertoire actuel: {current_dir}")
    
    # Vérification des fichiers essentiels
    essential_files = [
        'secret_envs_wrapper.py',
        'libs/',
        'environment_stats_analyzer.py'
    ]
    
    all_files_ok = True
    for file in essential_files:
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        print(f"   {status} {file}")
        if not exists:
            all_files_ok = False
    
    # 3. Vérification des bibliothèques natives
    print("\n3. 📚 VÉRIFICATION DES BIBLIOTHÈQUES NATIVES:")
    libs_dir = 'libs'
    
    if not os.path.exists(libs_dir):
        print(f"   ❌ Répertoire {libs_dir} non trouvé")
        return False
    
    # Lister le contenu du répertoire libs
    try:
        libs_content = os.listdir(libs_dir)
        print(f"   📂 Contenu du répertoire libs: {libs_content}")
        
        # Vérifier la DLL appropriée pour ce système
        system = platform.system().lower()
        if system == "windows":
            expected_lib = "secret_envs.dll"
        elif system == "linux":
            expected_lib = "libsecret_envs.so"
        elif system == "darwin":
            if "intel" in platform.processor().lower():
                expected_lib = "libsecret_envs_intel_macos.dylib"
            else:
                expected_lib = "libsecret_envs.dylib"
        else:
            print(f"   ⚠️ Système non supporté: {system}")
            return False
        
        lib_path = os.path.join(libs_dir, expected_lib)
        lib_exists = os.path.exists(lib_path)
        status = "✅" if lib_exists else "❌"
        print(f"   {status} Bibliothèque attendue: {expected_lib}")
        
        if lib_exists:
            # Vérifier la taille du fichier
            file_size = os.path.getsize(lib_path)
            print(f"   📊 Taille: {file_size:,} bytes")
        else:
            all_files_ok = False
        
    except Exception as e:
        print(f"   ❌ Erreur lors de l'accès au répertoire libs: {e}")
        return False
    
    # 4. Test d'import des modules Python
    print("\n4. 🐍 VÉRIFICATION DES MODULES PYTHON:")
    
    modules_to_test = [
        'numpy',
        'pandas', 
        'matplotlib',
        'tqdm'
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} - Module manquant")
            all_files_ok = False
    
    # 5. Test d'importation des environnements
    print("\n5. 🎮 TEST D'IMPORTATION DES ENVIRONNEMENTS:")
    
    try:
        from secret_envs_wrapper import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3
        print("   ✅ Import des wrappers réussi")
        
        # Test de création d'un environnement
        envs_to_test = [
            ('SecretEnv0', SecretEnv0),
            ('SecretEnv1', SecretEnv1),
            ('SecretEnv2', SecretEnv2),
            ('SecretEnv3', SecretEnv3)
        ]
        
        for env_name, env_class in envs_to_test:
            try:
                env = env_class()
                print(f"   ✅ {env_name} - Création réussie")
                
                # Test des méthodes de base
                num_states = env.num_states()
                num_actions = env.num_actions()
                print(f"      📊 États: {num_states}, Actions: {num_actions}")
                
            except Exception as e:
                print(f"   ❌ {env_name} - Erreur: {e}")
                all_files_ok = False
                
    except Exception as e:
        print(f"   ❌ Erreur d'import: {e}")
        all_files_ok = False
    
    # 6. Résumé et recommandations
    print("\n6. 📋 RÉSUMÉ ET RECOMMANDATIONS:")
    
    if all_files_ok:
        print("   🎉 Tous les tests sont passés avec succès!")
        print("   ✅ Vous pouvez lancer l'analyse complète avec: python run_analysis.py")
    else:
        print("   ⚠️ Des problèmes ont été détectés:")
        
        # Recommandations basées sur les erreurs
        if not os.path.exists('secret_envs_wrapper.py'):
            print("   💡 Naviguez vers le répertoire game/secret_env/")
            print("      cd game/secret_env")
        
        if not os.path.exists(libs_dir):
            print("   💡 Le répertoire libs/ est manquant")
            print("      Vérifiez que vous avez tous les fichiers du projet")
        
        if 'lib_exists' in locals() and not lib_exists:
            print(f"   💡 La bibliothèque {expected_lib} est manquante")
            print("      Vérifiez que vous avez la bonne version pour votre OS")
    
    return all_files_ok

def fix_directory():
    """Essaie de corriger le répertoire de travail (legacy function)"""
    return navigate_to_correct_directory()

def main():
    """Fonction principale du diagnostic"""
    print("🔧 DIAGNOSTIC AUTOMATIQUE DES ENVIRONNEMENTS SECRETS")
    print("=" * 60)
    
    # Navigation automatique vers le bon répertoire
    if not navigate_to_correct_directory():
        print("\n❌ Impossible de localiser les fichiers nécessaires")
        print("💡 Solutions possibles:")
        print("   1. Naviguez manuellement: cd game/secret_env")
        print("   2. Lancez depuis le répertoire RLProject")
        print("   3. Vérifiez que le projet est complet")
        return
    
    # Lancer le diagnostic complet
    success = diagnostic_complete()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ DIAGNOSTIC TERMINÉ: SYSTÈME PRÊT")
        print("🚀 Vous pouvez maintenant lancer:")
        print("   python run_analysis.py")
        print("   ou")
        print("   python quick_demo.py")
    else:
        print("❌ DIAGNOSTIC TERMINÉ: PROBLÈMES DÉTECTÉS")
        print("🔧 Suivez les recommandations ci-dessus")
    print("=" * 50)

if __name__ == "__main__":
    main() 