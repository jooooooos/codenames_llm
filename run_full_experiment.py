"""
Full Persona Experiment Runner
Runs all combinations of personas (21x21 = 441 combinations) with 5 games each
"""

import json
import os
from datetime import datetime
from itertools import product
import time

def run_full_persona_experiment(
    bnch,
    config,
    shared_llm_instance,
    num_games_per_combination=5,
    experiment_name=None,
    shared_persona=False,
    results_dir="experiment_results"
):
    """
    Run full persona experiment: all combinations of (None, "1", ..., "20") x (None, "1", ..., "20")

    Args:
        bnch: CodeNamesBenchmark instance
        config: Model configuration dict
        shared_llm_instance: Shared LLM instance for both agents
        num_games_per_combination: Number of games to run per persona pair (default: 5)
        experiment_name: Optional name for this experiment (auto-generated if None)
        shared_persona: Whether agents know each other's personas
        results_dir: Directory to save results (default: "experiment_results")

    Returns:
        dict: Complete experiment results with all combinations
    """

    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = f"full_persona_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    experiment_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Define all persona IDs (None + "1" through "20")
    persona_ids = [None] + [str(i) for i in range(1, 21)]

    # Calculate total combinations
    total_combinations = len(persona_ids) * len(persona_ids)
    total_games = total_combinations * num_games_per_combination

    print(f"\n{'='*80}")
    print(f"STARTING FULL PERSONA EXPERIMENT: {experiment_name}")
    print(f"{'='*80}")
    print(f"Model: {config['model_name']}")
    print(f"Persona IDs: {len(persona_ids)} (None + 1-20)")
    print(f"Combinations: {total_combinations} ({len(persona_ids)} x {len(persona_ids)})")
    print(f"Games per combination: {num_games_per_combination}")
    print(f"Total games: {total_games}")
    print(f"Persona sharing: {'Enabled' if shared_persona else 'Disabled'}")
    print(f"Results directory: {experiment_dir}")
    print(f"{'='*80}\n")

    # Metadata for the entire experiment
    experiment_metadata = {
        "experiment_name": experiment_name,
        "start_time": datetime.now().isoformat(),
        "model_config": {
            "model_name": config["model_name"],
            "model_type": config.get("type"),
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens")
        },
        "experiment_parameters": {
            "num_games_per_combination": num_games_per_combination,
            "total_combinations": total_combinations,
            "total_games": total_games,
            "shared_persona": shared_persona,
            "persona_ids": persona_ids
        },
        "combinations_completed": 0,
        "games_completed": 0
    }

    # Save initial metadata
    metadata_file = os.path.join(experiment_dir, "experiment_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(experiment_metadata, f, indent=2)

    # Storage for all results
    all_results = []

    # Track progress
    combination_count = 0
    games_count = 0
    experiment_start_time = time.time()

    # Iterate through all combinations
    for codemaster_persona, guesser_persona in product(persona_ids, persona_ids):
        combination_count += 1
        combination_start_time = time.time()

        # Create persona labels for display
        cm_label = codemaster_persona or "None"
        g_label = guesser_persona or "None"

        print(f"\n{'-'*80}")
        print(f"Combination {combination_count}/{total_combinations}: " +
              f"Codemaster={cm_label}, Guesser={g_label}")
        print(f"{'-'*80}")

        # Import here to ensure we get the updated module
        import llm_agent

        # Create fresh agents for this combination
        codemaster = llm_agent.LLMAgent(model_config=config, llm_instance=shared_llm_instance)
        codemaster.initialize_role('codemaster',
                                   persona_id=codemaster_persona,
                                   partner_persona_id=guesser_persona,
                                   shared_persona=shared_persona)

        guesser = llm_agent.LLMAgent(model_config=config, llm_instance=shared_llm_instance)
        guesser.initialize_role('guesser',
                               persona_id=guesser_persona,
                               partner_persona_id=codemaster_persona,
                               shared_persona=shared_persona)

        # Run games for this combination
        combination_id = f"cm{cm_label}_g{g_label}"

        try:
            results = bnch.run_collab_matchup(
                team_config=config,
                num_games=num_games_per_combination,
                codemaster=codemaster,
                guesser=guesser,
                codemaster_persona_id=codemaster_persona,
                guesser_persona_id=guesser_persona,
                persona_sharing=shared_persona,
                experiment_id=f"{experiment_name}_{combination_id}",
                save=True  # Save individual game files
            )

            # Add combination metadata
            results["combination_metadata"] = {
                "combination_number": combination_count,
                "combination_id": combination_id,
                "codemaster_persona_id": codemaster_persona,
                "guesser_persona_id": guesser_persona
            }

            all_results.append(results)
            games_count += num_games_per_combination

            # Print summary for this combination
            combination_duration = time.time() - combination_start_time
            print(f"\nCombination Summary:")
            print(f"  Win rate: {results['win_rate']:.1%}")
            print(f"  Avg turns: {results['average_turns']:.1f}")
            print(f"  Avg words/clue: {results['average_words_per_clue']:.2f}")
            print(f"  Duration: {combination_duration:.1f}s")

            # Save intermediate results after each combination
            combination_file = os.path.join(experiment_dir, f"combination_{combination_id}.json")
            with open(combination_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved: {combination_file}")

        except Exception as e:
            print(f"\n❌ ERROR in combination {combination_id}: {e}")
            # Log the error but continue
            error_result = {
                "combination_metadata": {
                    "combination_number": combination_count,
                    "combination_id": combination_id,
                    "codemaster_persona_id": codemaster_persona,
                    "guesser_persona_id": guesser_persona
                },
                "error": str(e),
                "status": "failed"
            }
            all_results.append(error_result)

            # Save error log
            error_file = os.path.join(experiment_dir, f"ERROR_{combination_id}.json")
            with open(error_file, "w") as f:
                json.dump(error_result, f, indent=2)

        # Update and save progress metadata
        experiment_metadata["combinations_completed"] = combination_count
        experiment_metadata["games_completed"] = games_count
        experiment_metadata["last_updated"] = datetime.now().isoformat()

        elapsed_time = time.time() - experiment_start_time
        avg_time_per_combo = elapsed_time / combination_count
        remaining_combos = total_combinations - combination_count
        estimated_remaining_time = avg_time_per_combo * remaining_combos

        experiment_metadata["progress"] = {
            "percent_complete": (combination_count / total_combinations) * 100,
            "elapsed_time_seconds": elapsed_time,
            "estimated_remaining_seconds": estimated_remaining_time,
            "estimated_total_seconds": elapsed_time + estimated_remaining_time
        }

        with open(metadata_file, "w") as f:
            json.dump(experiment_metadata, f, indent=2)

        # Progress update
        pct_complete = (combination_count / total_combinations) * 100
        print(f"\nProgress: {combination_count}/{total_combinations} ({pct_complete:.1f}%) | " +
              f"Elapsed: {elapsed_time/60:.1f}m | " +
              f"Est. remaining: {estimated_remaining_time/60:.1f}m")

    # Experiment complete
    total_duration = time.time() - experiment_start_time

    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"Total combinations: {combination_count}")
    print(f"Total games: {games_count}")
    print(f"Total duration: {total_duration/60:.1f} minutes ({total_duration/3600:.2f} hours)")
    print(f"Results saved to: {experiment_dir}")
    print(f"{'='*80}\n")

    # Save final aggregated results
    final_results = {
        "experiment_metadata": experiment_metadata,
        "all_combinations": all_results,
        "summary_statistics": {
            "total_combinations": combination_count,
            "total_games": games_count,
            "total_duration_seconds": total_duration,
            "average_duration_per_combination": total_duration / combination_count,
            "average_duration_per_game": total_duration / games_count
        }
    }

    experiment_metadata["end_time"] = datetime.now().isoformat()
    experiment_metadata["total_duration_seconds"] = total_duration

    final_results_file = os.path.join(experiment_dir, "complete_results.json")
    with open(final_results_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"Complete results saved to: {final_results_file}")

    return final_results


def load_experiment_results(experiment_name, results_dir="experiment_results"):
    """
    Load results from a completed or in-progress experiment.

    Args:
        experiment_name: Name of the experiment directory
        results_dir: Base results directory

    Returns:
        dict: Experiment results
    """
    experiment_dir = os.path.join(results_dir, experiment_name)

    # Try to load complete results first
    complete_file = os.path.join(experiment_dir, "complete_results.json")
    if os.path.exists(complete_file):
        with open(complete_file, "r") as f:
            return json.load(f)

    # If not complete, load individual combination files
    print(f"Loading in-progress experiment from: {experiment_dir}")

    metadata_file = os.path.join(experiment_dir, "experiment_metadata.json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Load all combination files
    all_results = []
    for filename in sorted(os.listdir(experiment_dir)):
        if filename.startswith("combination_") and filename.endswith(".json"):
            with open(os.path.join(experiment_dir, filename), "r") as f:
                all_results.append(json.load(f))

    return {
        "experiment_metadata": metadata,
        "all_combinations": all_results,
        "status": "in_progress"
    }
