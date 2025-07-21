import tkinter as tk
from tkinter import messagebox, font
from PIL import Image, ImageTk # Pour gérer les images (nécessite Pillow: pip install Pillow)
import os

# Importe les fonctions logiques de main.py
# Assurez-vous que main.py est dans le même dossier ou dans le PYTHONPATH
from main import get_computer_choice, determine_winner, ROCK, PAPER, SCISSORS, CHOICES

class RPSGameGUI:
    def __init__(self, master):
        self.master = master
        master.title("Pierre-Feuille-Ciseaux")
        master.geometry("600x700") # Taille de la fenêtre ajustée
        master.configure(bg="#f0f0f0")

        # Polices
        self.title_font = font.Font(family="Helvetica", size=24, weight="bold")
        self.text_font = font.Font(family="Helvetica", size=12)
        self.bold_text_font = font.Font(family="Helvetica", size=12, weight="bold")
        self.button_font = font.Font(family="Helvetica", size=14, weight="bold")
        self.score_font = font.Font(family="Helvetica", size=16, weight="bold")

        # Variables de jeu
        self.player1_name = "Joueur 1"
        self.player2_name = "Ordinateur" # Par défaut
        self.is_vs_computer = True
        self.current_round = 1
        self.player1_score = 0
        self.player2_score = 0
        self.ai_choice_round1 = None
        self.player1_choice = None
        self.player2_choice = None

        # --- Écran de sélection du mode de jeu ---
        self.setup_mode_selection_screen()

    def clear_window(self):
        for widget in self.master.winfo_children():
            widget.destroy()

    def setup_mode_selection_screen(self):
        self.clear_window()
        self.master.configure(bg="#e0e0e0")

        title_label = tk.Label(self.master, text="Pierre-Feuille-Ciseaux", font=self.title_font, bg="#e0e0e0", pady=20)
        title_label.pack()

        mode_label = tk.Label(self.master, text="Choisissez le mode de jeu :", font=self.text_font, bg="#e0e0e0", pady=10)
        mode_label.pack()

        vs_computer_button = tk.Button(self.master, text="Joueur vs Ordinateur", font=self.button_font,
                                       command=lambda: self.start_game_setup(vs_computer=True), bg="#4CAF50", fg="white", padx=10, pady=5)
        vs_computer_button.pack(pady=10)

        vs_player_button = tk.Button(self.master, text="Joueur vs Joueur", font=self.button_font,
                                     command=lambda: self.start_game_setup(vs_computer=False), bg="#2196F3", fg="white", padx=10, pady=5)
        vs_player_button.pack(pady=10)
        
        # Placeholder pour le design responsive (non applicable directement à Tkinter comme sur le web)
        # Mais nous pouvons rendre l'interface plus flexible
        self.master.grid_rowconfigure(0, weight=1) 
        self.master.grid_columnconfigure(0, weight=1)

    def start_game_setup(self, vs_computer):
        self.is_vs_computer = vs_computer
        self.clear_window()
        self.master.configure(bg="#f0f0f0")

        tk.Label(self.master, text="Entrez le nom du Joueur 1:", font=self.text_font, bg="#f0f0f0").pack(pady=5)
        self.p1_name_entry = tk.Entry(self.master, font=self.text_font)
        self.p1_name_entry.pack(pady=5)
        self.p1_name_entry.insert(0, "Joueur 1")

        if not vs_computer:
            tk.Label(self.master, text="Entrez le nom du Joueur 2:", font=self.text_font, bg="#f0f0f0").pack(pady=5)
            self.p2_name_entry = tk.Entry(self.master, font=self.text_font)
            self.p2_name_entry.pack(pady=5)
            self.p2_name_entry.insert(0, "Joueur 2")

        start_button = tk.Button(self.master, text="Commencer la partie", font=self.button_font,
                                 command=self.initialize_game_screen, bg="#008CBA", fg="white")
        start_button.pack(pady=20)

    def initialize_game_screen(self):
        self.player1_name = self.p1_name_entry.get() if self.p1_name_entry.get() else "Joueur 1"
        if not self.is_vs_computer:
            self.player2_name = self.p2_name_entry.get() if self.p2_name_entry.get() else "Joueur 2"
        else:
            self.player2_name = "Ordinateur"
        
        self.current_round = 1
        self.player1_score = 0
        self.player2_score = 0
        self.ai_choice_round1 = None
        self.player1_choice = None
        self.player2_choice = None
        self.setup_game_interface()

    def setup_game_interface(self):
        self.clear_window()
        self.master.configure(bg="#f0f0f0")

        # --- Header: Titre et Manche --- 
        header_frame = tk.Frame(self.master, bg="#f0f0f0", pady=10)
        header_frame.pack(fill="x")
        self.round_label = tk.Label(header_frame, text=f"Manche {self.current_round}/2", font=self.title_font, bg="#f0f0f0")
        self.round_label.pack()

        # --- Scores --- 
        score_frame = tk.Frame(self.master, bg="#f0f0f0", pady=10)
        score_frame.pack(fill="x")

        self.p1_score_label = tk.Label(score_frame, text=f"{self.player1_name}: {self.player1_score}", font=self.score_font, bg="#f0f0f0", fg="#333")
        self.p1_score_label.pack(side="left", padx=20, expand=True)

        self.p2_score_label = tk.Label(score_frame, text=f"{self.player2_name}: {self.player2_score}", font=self.score_font, bg="#f0f0f0", fg="#333")
        self.p2_score_label.pack(side="right", padx=20, expand=True)

        # --- Zone de choix --- 
        self.choices_frame = tk.Frame(self.master, bg="#f0f0f0", pady=20)
        self.choices_frame.pack()

        self.instruction_label = tk.Label(self.choices_frame, text=f"{self.player1_name}, faites votre choix :", font=self.text_font, bg="#f0f0f0", pady=10)
        self.instruction_label.pack()

        # --- Images/Boutons de choix --- 
        images_path = "images" # Les images doivent être ici (ex: images/pierre.png, images/feuille.png, images/ciseaux.png)
        self.choice_buttons = [] 
        self.image_references = [] # Pour éviter que les images soient garbage-collected

        button_frame = tk.Frame(self.choices_frame, bg="#f0f0f0")
        button_frame.pack()

        for choice_name in CHOICES:
            try:
                # Tentative de chargement de l'image
                img_path = os.path.join(images_path, f"{choice_name.lower()}.png")
                if not os.path.exists(img_path):
                    # Si l'image n'existe pas, utiliser un bouton texte
                    # print(f"Avertissement : Image {img_path} non trouvée. Utilisation d'un bouton texte.")
                    raise FileNotFoundError # Force l'utilisation du bouton texte
                
                pil_image = Image.open(img_path)
                # Redimensionner pour s'adapter, exemple : 100x100
                pil_image = pil_image.resize((100, 100), Image.Resampling.LANCZOS)
                tk_image = ImageTk.PhotoImage(pil_image)
                self.image_references.append(tk_image) # Conserver une référence

                button = tk.Button(button_frame, image=tk_image, command=lambda c=choice_name: self.process_player_choice(c), 
                                   bg="#f0f0f0", relief="raised", borderwidth=2)
            except Exception as e: #FileNotFoundError ou autre erreur de Pillow
                # print(f"Erreur chargement image {choice_name}: {e}")
                button = tk.Button(button_frame, text=choice_name.capitalize(), font=self.button_font, 
                                   command=lambda c=choice_name: self.process_player_choice(c), 
                                   width=10, height=3, bg="#ddd", fg="#333")
            button.pack(side="left", padx=15, pady=10)
            self.choice_buttons.append(button)
        
        # --- Zone d'affichage des résultats de la manche --- 
        self.result_frame = tk.Frame(self.master, bg="#f0f0f0", pady=15)
        self.result_frame.pack(fill="x")

        self.p1_choice_label = tk.Label(self.result_frame, text="", font=self.text_font, bg="#f0f0f0", width=25)
        self.p1_choice_label.pack()
        self.p2_choice_label = tk.Label(self.result_frame, text="", font=self.text_font, bg="#f0f0f0", width=25)
        self.p2_choice_label.pack()
        self.round_winner_label = tk.Label(self.result_frame, text="", font=self.bold_text_font, bg="#f0f0f0", pady=10)
        self.round_winner_label.pack()
        
        self.next_action_button = tk.Button(self.master, text="Manche Suivante", font=self.button_font, 
                                           command=self.next_round_or_end_game, state=tk.DISABLED, bg="#FF9800", fg="white")
        self.next_action_button.pack(pady=20)
        
        self.update_turn_instruction()

    def update_turn_instruction(self):
        if self.is_vs_computer or self.player1_choice is None:
            self.instruction_label.config(text=f"{self.player1_name}, faites votre choix :")
        else: # JcJ et P1 a joué
            self.instruction_label.config(text=f"{self.player2_name}, faites votre choix :")

    def process_player_choice(self, choice):
        if not self.is_vs_computer and self.player1_choice is not None and self.player2_choice is None: # Tour du J2
            self.player2_choice = choice
            self.p2_choice_label.config(text=f"{self.player2_name} a choisi : {self.player2_choice.capitalize()}")
            self.play_the_round()
        elif self.player1_choice is None: # Tour du J1 (ou seul tour si vs IA)
            self.player1_choice = choice
            self.p1_choice_label.config(text=f"{self.player1_name} a choisi : {self.player1_choice.capitalize()}")
            if self.is_vs_computer:
                self.player2_choice = get_computer_choice(self.current_round, self.ai_choice_round1)
                if self.current_round == 1: # Sauvegarder le premier choix de l'IA pour la manche 2
                    self.ai_choice_round1 = self.player2_choice
                self.p2_choice_label.config(text=f"{self.player2_name} a choisi : {self.player2_choice.capitalize()}")
                self.play_the_round()
            else:
                self.update_turn_instruction() # Mettre à jour pour le tour de P2
                # Désactiver les boutons de P1 pour qu'il ne puisse plus jouer avant P2
                for btn in self.choice_buttons:
                    btn.config(state=tk.DISABLED) # Sera réactivé pour la prochaine manche

    def play_the_round(self):
        # Désactiver les boutons de choix pendant l'affichage du résultat
        for btn in self.choice_buttons:
            btn.config(state=tk.DISABLED)

        winner_key = determine_winner(self.player1_choice, self.player2_choice)

        if winner_key == "player1":
            self.round_winner_label.config(text=f"{self.player1_name} gagne la manche !", fg="#4CAF50")
            self.player1_score += 1
        elif winner_key == "player2":
            self.round_winner_label.config(text=f"{self.player2_name} gagne la manche !", fg="#F44336")
            self.player2_score += 1
        else:
            self.round_winner_label.config(text="Égalité dans cette manche !", fg="#FFC107")

        self.p1_score_label.config(text=f"{self.player1_name}: {self.player1_score}")
        self.p2_score_label.config(text=f"{self.player2_name}: {self.player2_score}")
        
        self.next_action_button.config(state=tk.NORMAL)
        if self.current_round == 2:
            self.next_action_button.config(text="Voir les Résultats Finaux")

    def next_round_or_end_game(self):
        self.current_round += 1
        if self.current_round > 2:
            self.show_final_results()
        else:
            self.player1_choice = None
            self.player2_choice = None
            self.p1_choice_label.config(text="")
            self.p2_choice_label.config(text="")
            self.round_winner_label.config(text="")
            self.round_label.config(text=f"Manche {self.current_round}/2")
            self.next_action_button.config(state=tk.DISABLED, text="Manche Suivante")
            for btn in self.choice_buttons:
                btn.config(state=tk.NORMAL) # Réactiver les boutons pour la nouvelle manche
            self.update_turn_instruction()

    def show_final_results(self):
        self.clear_window()
        self.master.configure(bg="#e0e0e0")

        tk.Label(self.master, text="Résultats Finaux", font=self.title_font, bg="#e0e0e0", pady=20).pack()
        tk.Label(self.master, text=f"{self.player1_name}: {self.player1_score} manche(s)", font=self.score_font, bg="#e0e0e0").pack(pady=5)
        tk.Label(self.master, text=f"{self.player2_name}: {self.player2_score} manche(s)", font=self.score_font, bg="#e0e0e0").pack(pady=5)

        final_message = ""
        if self.player1_score > self.player2_score:
            final_message = f"{self.player1_name} remporte la partie ! Félicitations !"
        elif self.player2_score > self.player1_score:
            final_message = f"{self.player2_name} remporte la partie !"
        else:
            final_message = "La partie est une égalité !"
        
        tk.Label(self.master, text=final_message, font=self.bold_text_font, bg="#e0e0e0", pady=15).pack()

        tk.Button(self.master, text="Rejouer", font=self.button_font, command=self.setup_mode_selection_screen, bg="#4CAF50", fg="white").pack(pady=10)
        tk.Button(self.master, text="Quitter", font=self.button_font, command=self.master.quit, bg="#f44336", fg="white").pack(pady=5)


if __name__ == '__main__':
    root = tk.Tk()
    game_gui = RPSGameGUI(root)
    # Centre la fenêtre
    root.eval('tk::PlaceWindow . center') 
    root.mainloop()

