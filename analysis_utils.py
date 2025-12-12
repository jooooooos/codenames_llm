"""
Unified game log loading utilities for Codenames LLM analysis.

Supports two formats:
1. Old format (gemma2 - full): One game per file
   - Filename: full_persona_exp_TIMESTAMP_cm{X}_g{Y}_game{N}.json
   - JSON: Game data at root level

2. New format (tailored_words_to_persona1): Multiple games per file
   - Filename: combination_cm{X}_g{Y}.json
   - JSON: Games in 'games_detail' array, persona IDs as explicit fields
"""

import json
import os
import glob
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np


def detect_log_format(logs_dir):
    """
    Auto-detect whether directory contains old or new format logs.

    Returns: 'new' or 'old'
    """
    logs_dir = Path(logs_dir)

    # Check for new format indicators
    for filepath in logs_dir.glob('*.json'):
        if filepath.name.startswith('combination_'):
            return 'new'
        if filepath.name in ('experiment_metadata.json', 'complete_results.json'):
            continue
        # Check JSON structure
        try:
            with open(filepath) as f:
                data = json.load(f)
            if 'games_detail' in data:
                return 'new'
            # Old format has game data at root level
            if 'turns' in data and 'won' in data:
                return 'old'
        except:
            continue

    return 'old'  # Default to old format


def _parse_persona_id(value):
    """
    Convert persona ID to int or None.

    Handles:
    - int: returns as-is
    - str: converts "1" -> 1, "None" -> None
    - None: returns None
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.lower() == 'none':
            return None
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _parse_filename_old(filename):
    """
    Parse old format filename.
    Pattern: full_persona_exp_TIMESTAMP_cm{X}_g{Y}_game{N}.json

    Returns: (cm_id, g_id, game_num) or (None, None, None) on failure
    """
    pattern = r'cm([^_]+)_g([^_]+)_game(\d+)\.json'
    match = re.search(pattern, filename)
    if match:
        cm_id = _parse_persona_id(match.group(1))
        g_id = _parse_persona_id(match.group(2))
        game_num = int(match.group(3))
        return cm_id, g_id, game_num
    return None, None, None


def _parse_filename_new(filename):
    """
    Parse new format filename.
    Pattern: combination_cm{X}_g{Y}.json

    Returns: (cm_id, g_id) or (None, None) on failure
    """
    pattern = r'combination_cm([^_]+)_g([^_.]+)\.json'
    match = re.search(pattern, filename)
    if match:
        cm_id = _parse_persona_id(match.group(1))
        g_id = _parse_persona_id(match.group(2))
        return cm_id, g_id
    return None, None


def _iterate_games(logs_dir, max_persona_id=None):
    """
    Generator that yields (cm_id, g_id, game_num, game_data) tuples.

    Auto-detects format and handles both:
    - Old: yields single game per file
    - New: yields each game from games_detail array

    Args:
        logs_dir: Directory containing game log files
        max_persona_id: Maximum persona ID to include (filters out higher IDs)

    Yields:
        (cm_id, g_id, game_num, game_data) tuples
    """
    logs_dir = Path(logs_dir)
    format_type = detect_log_format(logs_dir)

    for filepath in logs_dir.glob('*.json'):
        # Skip metadata files (new format only)
        if filepath.name in ('experiment_metadata.json', 'complete_results.json'):
            continue

        try:
            with open(filepath) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {filepath.name}: {e}")
            continue

        if format_type == 'new':
            # New format: explicit persona IDs + games_detail array
            cm_id = _parse_persona_id(data.get('codemaster_persona_id'))
            g_id = _parse_persona_id(data.get('guesser_persona_id'))

            # Filter by max_persona_id
            if max_persona_id is not None:
                if (cm_id is not None and cm_id > max_persona_id) or \
                   (g_id is not None and g_id > max_persona_id):
                    continue

            for game in data.get('games_detail', []):
                game_num = game.get('game_id', 0)
                yield (cm_id, g_id, game_num, game)
        else:
            # Old format: parse filename, single game at root
            cm_id, g_id, game_num = _parse_filename_old(filepath.name)

            if cm_id is None and g_id is None and game_num is None:
                # Try alternative pattern (no game number)
                pattern = r'cm([^_]+)_g([^_.]+)\.json'
                match = re.search(pattern, filepath.name)
                if match:
                    cm_id = _parse_persona_id(match.group(1))
                    g_id = _parse_persona_id(match.group(2))
                    game_num = 0
                else:
                    continue

            # Filter by max_persona_id
            if max_persona_id is not None:
                if (cm_id is not None and cm_id > max_persona_id) or \
                   (g_id is not None and g_id > max_persona_id):
                    continue

            yield (cm_id, g_id, game_num, data)


def load_game_logs(logs_dir, max_persona_id=None):
    """
    Load all game logs and aggregate by persona pair.

    Works with both old and new formats via auto-detection.

    Args:
        logs_dir: Directory containing game log files
        max_persona_id: Maximum persona ID to include (filters out higher IDs)

    Returns:
        DataFrame with columns [cm_id, guesser_id, won, turns, duration, end_reason]
    """
    game_records = []

    format_type = detect_log_format(logs_dir)
    print(f"Detected format: {format_type}")

    for cm_id, g_id, game_num, game_data in _iterate_games(logs_dir, max_persona_id):
        game_records.append({
            'cm_id': cm_id,
            'guesser_id': g_id,
            'game_num': game_num,
            'won': game_data.get('won', False),
            'turns': game_data.get('total_turns', 0),
            'duration': game_data.get('game_duration', 0),
            'end_reason': game_data.get('end_reason', 'unknown')
        })

    df = pd.DataFrame(game_records)

    print(f"\nLoaded {len(df)} game records")
    if len(df) > 0:
        persona_pairs = df[(df['cm_id'].notna()) & (df['guesser_id'].notna())]
        print(f"Persona pairs (excluding None): {persona_pairs.groupby(['cm_id', 'guesser_id']).size().shape[0]}")
        baseline = df[(df['cm_id'].isna()) & (df['guesser_id'].isna())]
        print(f"Baseline (both None): {len(baseline)}")

    return df


def load_game_logs_with_turns(logs_dir, max_persona_id=None):
    """
    Load all game logs with full turn-level data.

    Works with both old and new formats via auto-detection.

    Args:
        logs_dir: Directory containing game log files
        max_persona_id: Maximum persona ID to include

    Returns:
        games_df: DataFrame with game-level info
        turns_df: DataFrame with turn-level info
    """
    game_records = []
    turn_records = []

    format_type = detect_log_format(logs_dir)
    print(f"Detected format: {format_type}")

    for cm_id, g_id, game_num, game_data in _iterate_games(logs_dir, max_persona_id):
        game_id = f"{cm_id}_{g_id}_{game_num}"

        # Game-level record
        game_records.append({
            'game_id': game_id,
            'cm_id': cm_id,
            'guesser_id': g_id,
            'game_num': game_num,
            'won': game_data.get('won', False),
            'total_turns': game_data.get('total_turns', 0),
            'end_reason': game_data.get('end_reason', 'unknown'),
            'duration': game_data.get('game_duration', 0)
        })

        # Turn-level records
        for turn in game_data.get('turns', []):
            clue_number = turn.get('clue_number', 0)
            results = turn.get('results', [])
            guesses = turn.get('guesses', [])

            correct_guesses = sum(1 for r in results if r == 'team word')

            # Get collaboration signals
            collab = turn.get('collaboration_signals', {})
            turn_ended_early = collab.get('turn_ended_early', False)

            # Compute clue utilization
            clue_utilization = correct_guesses / clue_number if clue_number > 0 else 0

            # Determine turn outcome
            turn_outcome = 'neutral'
            if 'assassin' in results:
                turn_outcome = 'assassin'
            elif results and results[-1] == 'team word':
                turn_outcome = 'all_correct'
            elif 'neutral' in results:
                turn_outcome = 'neutral'

            turn_records.append({
                'game_id': game_id,
                'cm_id': cm_id,
                'guesser_id': g_id,
                'turn_number': turn.get('turn_number', 0),
                'clue_word': turn.get('clue_word', ''),
                'clue_number': clue_number,
                'num_guesses': len(guesses),
                'correct_guesses': correct_guesses,
                'clue_utilization': clue_utilization,
                'turn_ended_early': turn_ended_early,
                'turn_outcome': turn_outcome,
                'turn_efficiency': turn.get('turn_efficiency', clue_utilization)
            })

    games_df = pd.DataFrame(game_records)
    turns_df = pd.DataFrame(turn_records)

    print(f"\nLoaded {len(games_df)} games, {len(turns_df)} turns")

    return games_df, turns_df


def load_turn_level_data(logs_dir, max_persona_id=None):
    """
    Load all game logs and extract turn-level data for cognitive endurance analysis.

    Only includes pairs where BOTH have personas (excludes baseline/partial).
    Skips turns with clue_number <= 0.

    Works with both old and new formats via auto-detection.

    Args:
        logs_dir: Directory containing game log files
        max_persona_id: Maximum persona ID to include

    Returns:
        turns_df: DataFrame with turn-level info including clue_utilization and cohort
    """
    turn_records = []

    format_type = detect_log_format(logs_dir)
    print(f"Detected format: {format_type}")

    for cm_id, g_id, game_num, game_data in _iterate_games(logs_dir, max_persona_id):
        # Only include pairs where BOTH have personas (exclude baseline/partial)
        if cm_id is None or g_id is None:
            continue

        game_id = f"{cm_id}_{g_id}_{game_num}"
        game_won = game_data.get('won', False)

        # Turn-level records
        for turn in game_data.get('turns', []):
            turn_number = turn.get('turn_number', 0)
            clue_number = turn.get('clue_number', 0)
            results = turn.get('results', [])

            # Skip turns with clue_number <= 0
            if clue_number <= 0:
                continue

            correct_guesses = sum(1 for r in results if r == 'team word')

            # Compute clue utilization
            clue_utilization = correct_guesses / clue_number

            # Determine if same persona
            is_same_persona = (cm_id == g_id)

            turn_records.append({
                'game_id': game_id,
                'cm_id': cm_id,
                'guesser_id': g_id,
                'game_num': game_num,
                'game_won': game_won,
                'turn_number': turn_number,
                'clue_word': turn.get('clue_word', ''),
                'clue_number': clue_number,
                'correct_guesses': correct_guesses,
                'clue_utilization': clue_utilization,
                'is_same_persona': is_same_persona,
                'cohort': 'Same Persona' if is_same_persona else 'Different Persona'
            })

    turns_df = pd.DataFrame(turn_records)

    print(f"\nLoaded {len(turns_df)} turns (with clue_number > 0)")

    return turns_df
