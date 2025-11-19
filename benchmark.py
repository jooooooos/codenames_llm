from typing import Dict, List, Tuple, Optional
from game_logger import GameLogger
from llm_agent import LLMAgent
import random
import json
import os
import time
from datetime import datetime

class CodeNamesBenchmark:
    def __init__(self, log_dir: str = "game_logs"):
        self.metrics = {}
        self.logger = GameLogger(log_dir)

    def simulate_game_collab(self, game_id: int, codemaster, guesser) -> Dict:
        """Simulate a collaborative game: 1 codemaster + 1 guesser, with detailed turn logs."""

        board = self.generate_board()
        team_words, _, neutral_words, assassin = self.split_words(board)
        random.shuffle(board)

        # Store initial board state for analysis
        initial_board = board.copy()
        initial_team_words = team_words.copy()
        initial_neutral_words = neutral_words.copy()
        initial_assassin = assassin

        game_state = {
            "guessed_words": set(),
            "current_turn_guesses": [],
            "guesses_remaining": 0,
            "past_turns": []
        }

        turn_count = 0
        game_over = False
        winner = None
        end_reason = None
        team_metrics = {"correct_guesses": 0, "incorrect_guesses": 0, "total_clues": 0}
        game_start_time = time.time()

        print(f"\nStarting collaborative game {game_id}")
        self.display_board(board, game_state["guessed_words"])

        while not game_over and turn_count < 20:
            turn_count += 1
            print(f"\n=== Turn {turn_count} ===")
            print(f"Remaining words to guess: {', '.join(team_words)}")

            # Codemaster gives clue (with timing)
            clue_start_time = time.time()
            clue_data = codemaster.give_clue("Team", team_words, neutral_words, [], assassin, game_state)
            clue_generation_time = time.time() - clue_start_time

            clue_word = clue_data.get("clue")
            clue_number = clue_data.get("number")
            codemaster_reasoning = clue_data.get("reasoning", "")
            print(f"Codemaster's clue: {clue_word} {clue_number}")
            if codemaster_reasoning:
                print(f"(Reasoning: {codemaster_reasoning})")


            team_metrics["total_clues"] += 1
            remaining_guesses = clue_number + 1
            game_state["guesses_remaining"] = remaining_guesses
            turn_guesses = []
            turn_results = []
            guesser_reasoning_list = []
            guess_times = []
            target_words_snapshot = team_words.copy()  # What words were available when clue was given

            # Guesser makes guesses
            while remaining_guesses > 0 and not game_over:
                guess_start_time = time.time()
                guess_data = guesser.make_guess("Team", board, clue_word, clue_number, game_state)
                guess_time = time.time() - guess_start_time

                guess = guess_data.get("guess")
                guess_reasoning = guess_data.get("reasoning", "")
                game_state["current_turn_guesses"].append(guess)
                turn_guesses.append(guess)
                guesser_reasoning_list.append(guess_reasoning)
                guess_times.append(guess_time)
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
                        end_reason = "all_found"
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
                    end_reason = "assassin"
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

            # Compute turn-level metrics
            correct_guesses_this_turn = sum(1 for r in turn_results if r == "team word")
            total_guesses_this_turn = len(turn_guesses)
            turn_efficiency = correct_guesses_this_turn / total_guesses_this_turn if total_guesses_this_turn > 0 else 0
            turn_ended_early = len(turn_guesses) < clue_number

            # Build guess metadata
            guess_metadata = []
            for idx, (g, r) in enumerate(zip(turn_guesses, turn_results)):
                guess_metadata.append({
                    "guess_word": g,
                    "guess_position_in_sequence": idx + 1,
                    "was_successful": r == "team word"
                })

            # Record detailed turn data with enhanced metadata
            game_state["past_turns"].append({
                # Original fields
                "turn_number": turn_count,
                "team": "Team",
                "clue_word": clue_word,
                "clue_number": clue_number,
                "guesses": turn_guesses,
                "results": turn_results,
                "remaining_team_words": list(team_words),
                "remaining_neutral_words": list(neutral_words),
                "guessed_words_so_far": list(game_state["guessed_words"]),

                # Phase 1: Reasoning traces and timing
                "codemaster_reasoning": codemaster_reasoning,
                "guesser_reasoning_per_guess": guesser_reasoning_list,
                "clue_generation_time": clue_generation_time,
                "guess_generation_times": guess_times,

                # Phase 2: NLP metadata
                "clue_metadata": {
                    "target_words_for_clue": target_words_snapshot,
                },
                "guess_metadata": guess_metadata,
                "turn_efficiency": turn_efficiency,

                # Phase 2: Collaboration signals
                "collaboration_signals": {
                    "turn_ended_early": turn_ended_early,
                    "guesses_used_all_remaining": len(turn_guesses) == clue_number + 1,
                }
            })

            self.display_board(board, game_state["guessed_words"])

        # If loop ended without winner being set, game hit max turns
        if winner is None:
            winner = False
            end_reason = "max_turns"

        game_duration = time.time() - game_start_time

        # Return both metrics and turn-by-turn logs with game context
        return {
            "metrics": {
                "correct_guesses": team_metrics["correct_guesses"],
                "incorrect_guesses": team_metrics["incorrect_guesses"],
                "words_per_clue": (team_metrics["correct_guesses"] / team_metrics["total_clues"])
                                  if team_metrics["total_clues"] else 0,
                "won": winner
            },
            "turns": game_state["past_turns"],
            # Game context for analysis
            "game_context": {
                "game_id": game_id,
                "initial_board": initial_board,
                "initial_team_words": initial_team_words,
                "initial_neutral_words": initial_neutral_words,
                "assassin_word": initial_assassin,
                "total_turns": turn_count,
                "won": winner,
                "end_reason": end_reason,
                "game_duration": game_duration
            }
        }

    def run_collab_matchup(self,
                          team_config: Dict,
                          num_games: int,
                          codemaster,
                          guesser,
                          codemaster_persona_id: Optional[str] = None,
                          guesser_persona_id: Optional[str] = None,
                          persona_sharing: bool = False,
                          experiment_id: Optional[str] = None,
                          save: bool = False) -> Dict:
        """Run multiple collaborative games and summarize metrics.

        Args:
            team_config: Model configuration dictionary
            num_games: Number of games to run
            codemaster: Codemaster agent
            guesser: Guesser agent
            codemaster_persona_id: Optional persona ID for codemaster
            guesser_persona_id: Optional persona ID for guesser
            persona_sharing: Whether agents know each other's personas
            experiment_id: Optional experiment ID (auto-generated if None)
            save: If True, save detailed game logs to JSON files
        """
        # Generate experiment ID if not provided
        if experiment_id is None:
            experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        experiment_start_time = time.time()
        timestamp = experiment_start_time

        results = {
            # Experiment identification
            "experiment_id": experiment_id,
            "timestamp": timestamp,
            "codemaster_persona_id": codemaster_persona_id,
            "guesser_persona_id": guesser_persona_id,
            "persona_sharing_enabled": persona_sharing,

            # Model configuration
            "model": team_config["model_name"],
            "experiment_metadata": {
                "model_temperature": team_config.get("temperature", 0.7),
                "model_max_tokens": team_config.get("max_tokens"),
                "experiment_date": datetime.fromtimestamp(timestamp).isoformat(),
                "model_type": team_config.get("type"),
            },

            # Aggregate metrics
            "games_played": 0,
            "games_won": 0,
            "total_correct_guesses": 0,
            "total_incorrect_guesses": 0,
            "average_words_per_clue": [],
            "average_turns": [],

            # Detailed per-game data
            "games_detail": [],
            "all_turns": []  # Backward compatibility: stores detailed logs for every game
        }

        for i in range(num_games):
            game_results = self.simulate_game_collab(i, codemaster, guesser)
            metrics = game_results["metrics"]
            game_context = game_results["game_context"]

            # Update aggregate metrics
            results["games_played"] += 1
            results["games_won"] += 1 if metrics["won"] else 0
            results["total_correct_guesses"] += metrics["correct_guesses"]
            results["total_incorrect_guesses"] += metrics["incorrect_guesses"]
            results["average_words_per_clue"].append(metrics["words_per_clue"])
            results["average_turns"].append(game_context["total_turns"])
            results["all_turns"].append(game_results["turns"])

            # Store detailed game data
            game_detail = {
                **game_context,
                "turns": game_results["turns"]
            }
            results["games_detail"].append(game_detail)

            # Save individual game to file if requested
            if save:
                self._save_game_to_file(experiment_id, i, game_detail)

        # Compute final averages
        results["win_rate"] = results["games_won"] / results["games_played"]
        results["average_words_per_clue"] = sum(results["average_words_per_clue"]) / len(results["average_words_per_clue"])
        results["average_turns"] = sum(results["average_turns"]) / len(results["average_turns"])
        results["experiment_duration"] = time.time() - experiment_start_time

        self.metrics = results
        return results

    def _save_game_to_file(self, experiment_id: str, game_id: int, game_detail: Dict):
        """Save individual game data to a JSON file."""
        os.makedirs(self.logger.log_dir, exist_ok=True)
        filename = os.path.join(self.logger.log_dir, f"{experiment_id}_game{game_id}.json")
        with open(filename, "w") as f:
            json.dump(game_detail, f, indent=2)
        print(f"Game {game_id} saved to {filename}")

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
