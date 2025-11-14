# main.py

import os
from dotenv import load_dotenv
from benchmark import CodeNamesBenchmark
import json

def main():
    # Load environment variables
    load_dotenv()
    
    # Model configurations
    model_configs = {
        "gpt4": {
            "type": "openai",
            "model_name": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.7
        },
        "gpt3": {
            "type": "openai",
            "model_name": "gpt-3.5-turbo",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.7
        },
        "gemini": {
            "type": "gemini",
            "model_name": "gemini-2.0-flash-exp",
            "api_key": os.getenv("GEMINI_API_KEY"),
            "temperature": 0.7
        },
        "claude": {
            "type": "claude",
            "model_name": "claude-3-opus-20240229",
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "temperature": 0.7
        }
    }

    # Initialize benchmark
    benchmark = CodeNamesBenchmark(log_dir="game_logs")
    
    # Choose which models to play against each other
    team_a = "gpt4"
    team_b = "gemini"
    
    try:
        print(f"\n=== Starting 2v2 Match ===")
        print(f"Team A ({team_a}): Codemaster + Guesser")
        print(f"Team B ({team_b}): Codemaster + Guesser")
        
        metrics = benchmark.run_matchup(
            team_a_config=model_configs[team_a],
            team_b_config=model_configs[team_b],
            num_games=3
        )

        # Print results
        print("\n=== Match Results ===")
        for team, stats in metrics.items():
            print(f"\n{team.upper()}:")
            print(f"Wins: {stats['wins']}/{stats['games_played']}")
            print(f"Win Rate: {stats['win_rate']:.2%}")
            print(f"Correct guesses: {stats['total_correct_guesses']}")
            print(f"Incorrect guesses: {stats['total_incorrect_guesses']}")
            print(f"Average words per clue: {stats['average_words_per_clue']:.2f}")

    except Exception as e:
        print(f"Match failed: {e}")
        raise e

if __name__ == "__main__":
    main()