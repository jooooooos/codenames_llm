# main.py

import os
from dotenv import load_dotenv
from benchmark import CodeNamesBenchmark
from llm_providers import create_llm
import llm_agent

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
        },
        # Local HuggingFace models
        "llama31": {
            "type": "local_hf",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "temperature": 0.7,
            # "load_in_4bit": True
        },
        "qwen3": {
            "type": "local_hf",
            "model_name": "Qwen/Qwen3-8B",
            "temperature": 0.7,
            "max_tokens": 2048,  # Higher for Qwen's reasoning mode
            "disable_reasoning": True,  # Disable <think> tags
            # "load_in_4bit": True
        },
        "mistral": {
            "type": "local_hf",
            "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
            "temperature": 0.7,
            # "load_in_4bit": True
        }
    }

    # Initialize benchmark
    benchmark = CodeNamesBenchmark(log_dir="game_logs")

    # Choose which model to use for collaborative game
    model_choice = "llama31"  # Change this to test different models

    # ===== PERSONA CONFIGURATION =====
    # Set persona IDs for codemaster and guesser (use None for no persona)
    # Available persona IDs: "1" through "20" (see personas.json for details)
    #
    # Examples of the 4 combinations:
    # 1. No personas (baseline):
    #    codemaster_persona, guesser_persona = None, None
    #
    # 2. Only codemaster has persona:
    #    codemaster_persona, guesser_persona = "1", None
    #
    # 3. Only guesser has persona:
    #    codemaster_persona, guesser_persona = None, "2"
    #
    # 4. Both have personas (can be same or different):
    #    codemaster_persona, guesser_persona = "1", "3"

    codemaster_persona = None  # Change to "1", "2", etc. to enable persona
    guesser_persona = None     # Change to "1", "2", etc. to enable persona

    # ===== PERSONA SHARING TOGGLE =====
    # When True: agents know each other's personas and can adapt their communication
    # When False (default): personas remain isolated, agents unaware of partner's background
    shared_persona = False  # Change to True to enable persona sharing

    try:
        print(f"\n=== Starting Collaborative Codenames ===")
        print(f"Model: {model_configs[model_choice]['model_name']}")
        print(f"Codemaster Persona: {codemaster_persona or 'None (baseline)'}")
        print(f"Guesser Persona: {guesser_persona or 'None (baseline)'}")
        print(f"Persona Sharing: {'Enabled' if shared_persona else 'Disabled'}")
        print(f"Mode: Codemaster + Guesser vs. Board")
        print(f"Games: 3\n")

        # Get the model config
        config = model_configs[model_choice]

        # Create shared LLM instance for both agents
        shared_llm = create_llm(config)

        # Create codemaster and guesser agents with optional personas
        codemaster = llm_agent.LLMAgent(model_config=config, llm_instance=shared_llm)
        codemaster.initialize_role('codemaster',
                                  persona_id=codemaster_persona,
                                  partner_persona_id=guesser_persona,
                                  shared_persona=shared_persona)

        guesser = llm_agent.LLMAgent(model_config=config, llm_instance=shared_llm)
        guesser.initialize_role('guesser',
                               persona_id=guesser_persona,
                               partner_persona_id=codemaster_persona,
                               shared_persona=shared_persona)

        # Run collaborative games
        results = benchmark.run_collab_matchup(
            team_config=config,
            num_games=3,
            codemaster=codemaster,
            guesser=guesser
        )

        # Print results
        print("\n=== Game Results ===")
        print(f"Model: {results['model']}")
        print(f"Games played: {results['games_played']}")
        print(f"Wins: {results['games_won']}")
        print(f"Win rate: {results['win_rate']:.1%}")
        print(f"Average turns per game: {results['average_turns']:.1f}")
        print(f"Total correct guesses: {results['total_correct_guesses']}")
        print(f"Total incorrect guesses: {results['total_incorrect_guesses']}")
        if results['total_correct_guesses'] > 0:
            accuracy = results['total_correct_guesses'] / (results['total_correct_guesses'] + results['total_incorrect_guesses'])
            print(f"Guess accuracy: {accuracy:.1%}")

    except Exception as e:
        print(f"Game failed: {e}")
        raise e

if __name__ == "__main__":
    main()
