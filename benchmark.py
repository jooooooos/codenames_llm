from typing import Dict, List, Tuple
from game_logger import GameLogger
from llm_agent import LLMAgent
import random
import json
import os

class CodeNamesBenchmark:
    def __init__(self, log_dir: str = "game_logs"):
        self.metrics = {}
        self.logger = GameLogger(log_dir)

    def simulate_game_collab(self, game_id: int, codemaster, guesser) -> Dict:
        """Simulate a collaborative game: 1 codemaster + 1 guesser, with detailed turn logs."""

        board = self.generate_board()
        team_words, _, neutral_words, assassin = self.split_words(board)
        random.shuffle(board)
        game_state = {
            "guessed_words": set(),
            "current_turn_guesses": [],
            "guesses_remaining": 0,
            "past_turns": []
        }

        turn_count = 0
        game_over = False
        winner = None
        team_metrics = {"correct_guesses": 0, "incorrect_guesses": 0, "total_clues": 0}

        print(f"\nStarting collaborative game {game_id}")
        self.display_board(board, game_state["guessed_words"])

        while not game_over and turn_count < 20:
            turn_count += 1
            print(f"\n=== Turn {turn_count} ===")
            print(f"Remaining words to guess: {', '.join(team_words)}")

            # Codemaster gives clue
            clue_data = codemaster.give_clue("Team", team_words, neutral_words, [], assassin, game_state)
            clue_word = clue_data.get("clue")
            clue_number = clue_data.get("number")
            reasoning = clue_data.get("reasoning", "")
            print(f"Codemaster's clue: {clue_word} {clue_number}")
            if reasoning:
                print(f"(Reasoning: {reasoning})")


            team_metrics["total_clues"] += 1
            remaining_guesses = clue_number + 1
            game_state["guesses_remaining"] = remaining_guesses
            turn_guesses = []
            turn_results = []

            # Guesser makes guesses
            while remaining_guesses > 0 and not game_over:
                guess_data = guesser.make_guess("Team", board, clue_word, clue_number, game_state)
                guess = guess_data.get("guess")
                guess_reasoning = guess_data.get("reasoning", "")
                game_state["current_turn_guesses"].append(guess)
                turn_guesses.append(guess)
                print(f"Guesser's guess: {guess}")
                if guess_reasoning:
                    print(f"(Reasoning: {guess_reasoning})")


                if guess in team_words:
                    print("Correct guess!")
                    team_metrics["correct_guesses"] += 1
                    team_words.remove(guess)
                    game_state["guessed_words"].add(guess)
                    turn_results.append("team word")

                    if not team_words:
                        game_over = True
                        winner = True
                        print("All words guessed! You win!")
                        break
                elif guess in neutral_words:
                    print("Hit a neutral word.")
                    team_metrics["incorrect_guesses"] += 1
                    neutral_words.remove(guess)
                    game_state["guessed_words"].add(guess)
                    turn_results.append("neutral")
                    break
                elif guess == assassin:
                    print("Hit the assassin! Game over.")
                    team_metrics["incorrect_guesses"] += 1
                    turn_results.append("assassin")
                    game_over = True
                    winner = False
                    break
                else:
                    print("Guessed a word not on the board")
                    team_metrics["incorrect_guesses"] += 1
                    game_state["guessed_words"].add(guess)
                    turn_results.append("neutral")
                    break
                  

                remaining_guesses -= 1
                game_state["guesses_remaining"] = remaining_guesses
                if remaining_guesses > 0:
                    print(f"Remaining guesses this turn: {remaining_guesses}")

            # Record detailed turn data
            game_state["past_turns"].append({
                "turn_number": turn_count,
                "team": "Team",
                "clue_word": clue_word,
                "clue_number": clue_number,
                "guesses": turn_guesses,
                "results": turn_results,
                "remaining_team_words": list(team_words),
                "remaining_neutral_words": list(neutral_words),
                "guessed_words_so_far": list(game_state["guessed_words"])
            })

            self.display_board(board, game_state["guessed_words"])

        # Return both metrics and turn-by-turn logs
        return {
            "metrics": {
                "correct_guesses": team_metrics["correct_guesses"],
                "incorrect_guesses": team_metrics["incorrect_guesses"],
                "words_per_clue": (team_metrics["correct_guesses"] / team_metrics["total_clues"])
                                  if team_metrics["total_clues"] else 0,
                "won": winner
            },
            "turns": game_state["past_turns"]
        }

    def run_collab_matchup(self, team_config: Dict, num_games: int, codemaster, guesser) -> Dict:
        """Run multiple collaborative games and summarize metrics."""
        results = {
            "model": team_config["model_name"],
            "games_played": 0,
            "games_won": 0,
            "total_correct_guesses": 0,
            "total_incorrect_guesses": 0,
            "average_words_per_clue": [],
            "all_turns": []  # stores detailed logs for every game
        }

        for i in range(num_games):
            game_results = self.simulate_game_collab(i, codemaster, guesser)
            metrics = game_results["metrics"]
            results["games_played"] += 1
            results["games_won"] += 1 if metrics["won"] else 0
            results["total_correct_guesses"] += metrics["correct_guesses"]
            results["total_incorrect_guesses"] += metrics["incorrect_guesses"]
            results["average_words_per_clue"].append(metrics["words_per_clue"])
            results["all_turns"].append(game_results["turns"])

        results["win_rate"] = results["games_won"] / results["games_played"]
        results["average_words_per_clue"] = sum(results["average_words_per_clue"]) / len(results["average_words_per_clue"])

        self.metrics = results
        return results

    def generate_board(self) -> List[str]:
        with open("words/default.txt", "r") as file:
            words = file.read().splitlines()
        return random.sample(words, 25)

    def split_words(self, board: List[str]) -> Tuple[List[str], List[str], List[str], str]:
        random.shuffle(board)
        return board[:9], [], board[9:24], board[24]  # only team, neutral, and assassin

    def display_board(self, board: List[str], guessed_words: set):
        print("\nBoard State:")
        for i in range(5):
            row = board[i * 5:(i + 1) * 5]
            display = [f"[{word}]" if word in guessed_words else word for word in row]
            print(" | ".join(display))
        print()

    def export_turn_logs(self, filename: str = "turn_logs.json"):
        """
        Export detailed turn-by-turn logs of all games to a JSON file.
        Assumes `self.metrics['all_turns']` exists from run_collab_matchup.
        """
        if not hasattr(self, 'metrics') or 'all_turns' not in self.metrics:
            print("No turn logs available to export.")
            return

        os.makedirs(os.path.dirname(filename), exist_ok=True)  # create folder if needed
        with open(filename, "w") as f:
            json.dump(self.metrics['all_turns'], f, indent=4)
        
        print(f"Turn logs exported to {filename}")
