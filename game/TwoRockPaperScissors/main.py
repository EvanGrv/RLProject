import random

# Constantes pour les choix
ROCK = "pierre"
PAPER = "feuille"
SCISSORS = "ciseaux"
CHOICES = [ROCK, PAPER, SCISSORS]

PLAYER_CHOICE_MAP = {
    'p': ROCK,
    'f': PAPER,
    'c': SCISSORS
}

def get_player_choice_terminal(player_name):
    """Demande au joueur de choisir entre pierre, feuille, ciseaux via le terminal."""
    while True:
        choice_input = input(f"{player_name}, choisissez pierre, feuille, ou ciseaux (p, f, c): ").lower()
        if choice_input in PLAYER_CHOICE_MAP:
            return PLAYER_CHOICE_MAP[choice_input]
        else:
            print("Choix invalide. Veuillez réessayer.")

def get_computer_choice(round_number, previous_choice=None):
    """Détermine le choix de l'ordinateur."""
    if round_number == 1:
        return random.choice(CHOICES)
    else:
        return previous_choice

def determine_winner(choice1, choice2):
    """
    Détermine le gagnant d'une manche.
    Retourne "player1", "player2", ou "tie".
    """
    if choice1 == choice2:
        return "tie"
    elif (choice1 == ROCK and choice2 == SCISSORS) or \
         (choice1 == SCISSORS and choice2 == PAPER) or \
         (choice1 == PAPER and choice2 == ROCK):
        return "player1"
    else:
        return "player2"

def play_round_terminal(round_number, player1_name, player2_name, is_vs_computer, ai_previous_choice=None):
    """Joue une manche du jeu en mode terminal."""
    print(f"\n--- Manche {round_number} ---")

    choice_p1 = get_player_choice_terminal(player1_name)
    ai_current_choice = None # Sera le choix de l'IA pour cette manche

    if is_vs_computer:
        choice_p2 = get_computer_choice(round_number, ai_previous_choice)
        print(f"{player2_name} (ordinateur) a choisi : {choice_p2}")
        ai_current_choice = choice_p2
    else:
        choice_p2 = get_player_choice_terminal(player2_name)

    print(f"{player1_name} a choisi : {choice_p1}")
    if not is_vs_computer:
        print(f"{player2_name} a choisi : {choice_p2}")

    winner_key = determine_winner(choice_p1, choice_p2)
    
    if winner_key == "tie":
        print("C'est une égalité !")
    elif winner_key == "player1":
        print(f"{player1_name} gagne la manche !")
    else: # player2
        print(f"{player2_name} gagne la manche !")
        
    return winner_key, ai_current_choice, choice_p1, choice_p2

def run_terminal_game():
    """Fonction principale du jeu en mode terminal."""
    print("Bienvenue au jeu Pierre-Feuille-Ciseaux en deux manches ! (Mode Terminal)")

    while True:
        game_mode = input("Choisissez le mode de jeu (1 pour Joueur vs Ordinateur, 2 pour Joueur vs Joueur): ")
        if game_mode in ['1', '2']:
            break
        print("Mode de jeu invalide. Veuillez entrer 1 ou 2.")

    is_vs_computer = (game_mode == '1')

    player1_name = input("Nom du Joueur 1: ")
    player2_name = "Ordinateur" if is_vs_computer else input("Nom du Joueur 2: ")

    scores = {"player1": 0, "player2": 0, "tie": 0} # Ajout de "tie" pour la clarté, même si non utilisé pour le score final du joueur
    ai_choice_round1 = None # Choix de l'IA à la manche 1, à réutiliser en manche 2

    # Manche 1
    winner_key_r1, ai_choice_round1, _, _ = play_round_terminal(1, player1_name, player2_name, is_vs_computer)
    if winner_key_r1 == "player1":
        scores["player1"] += 1
    elif winner_key_r1 == "player2":
        scores["player2"] += 1
    # Pas de points pour "tie" dans le score des joueurs

    # Manche 2
    # ai_choice_round1 (le choix de l'IA à la manche 1) est passé comme ai_previous_choice
    winner_key_r2, _, _, _ = play_round_terminal(2, player1_name, player2_name, is_vs_computer, ai_choice_round1)
    if winner_key_r2 == "player1":
        scores["player1"] += 1
    elif winner_key_r2 == "player2":
        scores["player2"] += 1

    # Affichage des résultats finaux
    print("\n--- Résultats Finaux ---")
    print(f"{player1_name}: {scores['player1']} manche(s)")
    print(f"{player2_name}: {scores['player2']} manche(s)")

    if scores['player1'] > scores['player2']:
        print(f"{player1_name} remporte la partie !")
    elif scores['player2'] > scores['player1']:
        print(f"{player2_name} remporte la partie !")
    else:
        print("La partie est une égalité !")

if __name__ == "__main__":
    # Par défaut, lance le jeu en mode terminal.
    # Pour lancer la GUI, il faudra exécuter interface.py
    run_terminal_game()
