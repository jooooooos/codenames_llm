import random

class CodenamesGame:
    def __init__(self):
        self.board = self.generate_board()
        self.teams = {
            'Team A': {'Words': [], 'Guessed': []},
            'Team B': {'Words': [], 'Guessed': []}
        }
        self.assassin = None
        self.neutral_words = []
        self.current_turn = 'Team A'
        self.game_over = False
        self.winner = None
    
    def generate_board(self):
        words = ["apple", "banana", "satellite", "river", "mountain", "cell", "lion", "rocket", "cake", "cloud",
                 "piano", "laser", "ocean", "engine", "guitar", "book", "moon", "camera", "castle", "space",
                 "robot", "forest", "train", "garden", "keyboard"]
        random.shuffle(words)
        return words[:25]

    def assign_words(self):
        all_words = self.board.copy()
        self.teams['Team A']['Words'] = random.sample(all_words, 8)
        all_words = [word for word in all_words if word not in self.teams['Team A']['Words']]
        self.teams['Team B']['Words'] = random.sample(all_words, 7)
        all_words = [word for word in all_words if word not in self.teams['Team B']['Words']]
        
        self.assassin = random.choice(all_words)
        all_words.remove(self.assassin)
        
        self.neutral_words = all_words

    def switch_turn(self):
        self.current_turn = 'Team B' if self.current_turn == 'Team A' else 'Team A'

    def get_clue(self, llm_instance):
        return llm_instance.give_clue(
            self.current_turn,
            self.teams[self.current_turn]['Words'],
            self.neutral_words,
            self.assassin
        )
    
    def make_guess(self, llm_instance, clue):
        team = self.current_turn
        guess = llm_instance.guess_word(team, clue, self.board)
        print(f"{team} guesses: {guess}")
        
        if guess == self.assassin:
            self.game_over = True
            self.winner = 'Team B' if team == 'Team A' else 'Team A'
            return "assassin"
        elif guess in self.teams[team]['Words']:
            self.teams[team]['Words'].remove(guess)
            self.teams[team]['Guessed'].append(guess)
            return True
        elif guess in self.neutral_words:
            self.neutral_words.remove(guess)
            return False
        elif guess in self.teams['Team A' if team == 'Team B' else 'Team B']['Words']:
            other_team = 'Team A' if team == 'Team B' else 'Team B'
            self.teams[other_team]['Words'].remove(guess)
            self.teams[other_team]['Guessed'].append(guess)
            return False
        else:
            return False

    def check_win_condition(self):
        for team, data in self.teams.items():
            if not data['Words']:
                print(f"{team} has guessed all their words. They win!")
                self.game_over = True
                self.winner = team
                return True
        return False