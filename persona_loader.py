"""
Persona loader utility for Codenames LLM agents.

This module loads persona profiles from personas.json and provides
utilities for retrieving persona text to inject into agent prompts.
"""

import json
import os
from typing import Dict, List, Optional


# Cache for loaded personas
_PERSONAS_CACHE: Optional[Dict[str, str]] = None


def load_personas(filepath: str = "personas.json") -> Dict[str, str]:
    """
    Load all personas from the JSON file.

    Args:
        filepath: Path to the personas JSON file (default: "personas.json")

    Returns:
        Dictionary mapping persona IDs to persona text

    Raises:
        FileNotFoundError: If personas.json doesn't exist
        json.JSONDecodeError: If personas.json is invalid JSON
    """
    global _PERSONAS_CACHE

    # Return cached personas if already loaded
    if _PERSONAS_CACHE is not None:
        return _PERSONAS_CACHE

    # Resolve path relative to this file's directory
    if not os.path.isabs(filepath):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        _PERSONAS_CACHE = json.load(f)

    return _PERSONAS_CACHE


def get_persona(persona_id: str, filepath: str = "personas.json") -> str:
    """
    Retrieve a specific persona's text by ID.

    Args:
        persona_id: The persona ID (e.g., "1", "2", "3")
        filepath: Path to the personas JSON file (default: "personas.json")

    Returns:
        The persona text (narrative description)

    Raises:
        KeyError: If the persona_id doesn't exist
        FileNotFoundError: If personas.json doesn't exist
    """
    personas = load_personas(filepath)

    if persona_id not in personas:
        available = list_persona_ids(filepath)
        raise KeyError(
            f"Persona ID '{persona_id}' not found. "
            f"Available IDs: {', '.join(available)}"
        )

    return personas[persona_id]


def list_persona_ids(filepath: str = "personas.json") -> List[str]:
    """
    List all available persona IDs.

    Args:
        filepath: Path to the personas JSON file (default: "personas.json")

    Returns:
        List of persona IDs as strings
    """
    personas = load_personas(filepath)
    return sorted(personas.keys(), key=lambda x: int(x))


def clear_cache():
    """Clear the personas cache. Useful for reloading after file changes."""
    global _PERSONAS_CACHE
    _PERSONAS_CACHE = None
