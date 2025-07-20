#!/usr/bin/env python3
"""
🚀 LANCEUR UNIVERSEL POUR L'ANALYSE DES ENVIRONNEMENTS SECRETS

Ce script peut être lancé depuis n'importe où dans le projet RLProject.
Il va automatiquement naviguer vers le bon répertoire et lancer l'analyse.
"""

import os
import sys
import subprocess

def find_project_root():
    """Trouve la racine du projet RLProject"""
    current_path = os.getcwd()
    
    # Chercher vers le haut jusqu'à trouver RLProject
    while current_path and current_path != os.path.dirname(current_path):
        if 'RLProject' in os.path.basename(current_path):
            return current_path
        
        # Vérifier si on trouve des indicateurs du projet
        if os.path.exists(os.path.join(current_path, 'game', 'secret_env', 'secret_envs_wrapper.py')):
            return current_path
            
        current_path = os.path.dirname(current_path)
    
    return None

def main():
    """Fonction principale du lanceur"""
    print("🚀 LANCEUR UNIVERSEL D'ANALYSE DES ENVIRONNEMENTS SECRETS")
    print("=" * 60)
    
    # Trouver la racine du projet
    project_root = find_project_root()
    
    if not project_root:
        print("❌ Impossible de trouver le projet RLProject")
        print("💡 Assurez-vous d'être dans un répertoire du projet")
        return
    
    secret_env_path = os.path.join(project_root, 'game', 'secret_env')
    
    if not os.path.exists(secret_env_path):
        print(f"❌ Répertoire secret_env non trouvé: {secret_env_path}")
        return
    
    print(f"✅ Projet trouvé: {project_root}")
    print(f"✅ Répertoire secret_env: {secret_env_path}")
    
    # Naviguer vers le répertoire
    os.chdir(secret_env_path)
    print(f"📁 Navigation vers: {secret_env_path}")
    
    # Menu d'options
    print("\n🎯 QUE VOULEZ-VOUS FAIRE ?")
    print("1. 🔧 Diagnostic complet")
    print("2. 🎮 Test rapide des environnements")
    print("3. 📊 Analyse complète avec statistiques")
    print("4. 📝 Voir les exemples d'utilisation")
    print("0. ❌ Quitter")
    
    try:
        choice = input("\n👉 Votre choix (0-4): ").strip()
        
        if choice == '1':
            print("\n🔧 Lancement du diagnostic...")
            subprocess.run([sys.executable, 'diagnostic.py'])
            
        elif choice == '2':
            print("\n🎮 Lancement du test rapide...")
            subprocess.run([sys.executable, 'quick_demo.py'])
            
        elif choice == '3':
            print("\n📊 Lancement de l'analyse complète...")
            subprocess.run([sys.executable, 'run_analysis.py'])
            
        elif choice == '4':
            print("\n📝 Lancement des exemples...")
            subprocess.run([sys.executable, 'example_usage.py'])
            
        elif choice == '0':
            print("👋 Au revoir!")
            
        else:
            print("❌ Choix invalide. Utilisez 0, 1, 2, 3 ou 4.")
            
    except KeyboardInterrupt:
        print("\n👋 Arrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main() 