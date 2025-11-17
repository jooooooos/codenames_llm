# llm_agent.py

from typing import Dict, List, Optional, Any
from llm_providers import create_llm
import time  # Added this import

import json
import re
from prompts import (
    CODEMASTER_SYSTEM_PROMPT,
    GUESSER_SYSTEM_PROMPT,
    get_codemaster_prompt,
    get_guesser_prompt,
    GUESSER_FEEDBACK_PROMPT
)
from persona_loader import get_persona

class LLMAgent:
    def __init__(self, model_config: Dict, llm_instance: Optional[Any] = None):
        """Initialize an LLM agent with specific configuration"""
        self.model_config = model_config  # Store config for later use
        if llm_instance is not None:
            self.llm = llm_instance
        else:
            # Fallback/Original behavior: load the model from config
            self.llm = create_llm(model_config)
        self.role: Optional[str] = None

    def initialize_role(self, role: str, persona_id: Optional[str] = None):
        """
        Set the role for this LLM agent, optionally with a persona.

        Args:
            role: Either 'codemaster' or 'guesser'
            persona_id: Optional persona ID (e.g., "1", "2", "3") from personas.json
        """
        self.role = role
        base_prompt = (CODEMASTER_SYSTEM_PROMPT if role == 'codemaster'
                      else GUESSER_SYSTEM_PROMPT)

        # If persona is provided, prepend it to establish identity first
        if persona_id:
            persona_text = get_persona(persona_id)
            # Persona comes FIRST to establish who the agent is
            self.system_prompt = f"{persona_text}\n\n{base_prompt}"
        else:
            self.system_prompt = base_prompt

        # Add disable reasoning instruction if configured
        if self.model_config.get("disable_reasoning", False):
            self.system_prompt += "\n\nIMPORTANT: Output ONLY valid JSON. Do not use <think> tags, reasoning blocks, or explanations. Provide only the JSON response."

    def _make_request(self, messages: List[Dict], max_tokens: int) -> str:
        """Make an API request with retries"""
        max_retries = 5
        base_delay = 2.0  # Increased base delay
        
        for attempt in range(max_retries):
            try:
                return self.llm.generate(messages, max_tokens)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                
                # Exponential backoff with longer initial delay
                wait_time = base_delay * (2 ** attempt)
                print(f"API Error: {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

    def give_clue(self,
                  team: str,
                  team_words: List[str],
                  neutral_words: List[str],
                  avoid_words: List[str],
                  assassin: str,
                  game_state: Dict) -> Dict:
        """Generate a clue as the Codemaster (JSON output)"""
        if self.role != "codemaster":
            raise ValueError("This agent is not initialized as a Codemaster")

        prompt = get_codemaster_prompt(
            team=team,
            team_words=team_words,
            neutral_words=neutral_words,
            avoid_words=avoid_words,
            assassin=assassin,
            game_state=game_state
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Use configurable max_tokens, default to 200 for codemaster
        max_tokens = self.model_config.get("max_tokens", 200)
        raw = self._make_request(messages, max_tokens=max_tokens)

        all_board_words = {word.upper() for word in team_words + neutral_words + avoid_words + [assassin]}
        print("codemaster raw output: " + raw)
        try:
            json_match = None

            # 1. Check 1 (Most likely to succeed): Find content inside ```json{...}```
            json_match = re.search(r"```json\s*(\{.*?})\s*```", raw, re.DOTALL)

            if not json_match:
                # 2. Check 2: Find content inside ```{...}``` (missing 'json' tag)
                json_match = re.search(r"```\s*(\{.*?})\s*```", raw, re.DOTALL)
                
            if not json_match:
                # 3. Check 3 (Fallback): Find the first raw JSON object, ignoring all surrounding noise
                json_match = re.search(r"(\{.*?})", raw, re.DOTALL)
                
                if not json_match:
                    raise ValueError("Could not find any JSON-like object in the response.")

            # --- JSON Extraction ---
            # If groups > 0, we matched a code block (use group(1) for the content)
            # Otherwise, use group(0) for the raw JSON fallback
            if len(json_match.groups()) > 0:
                extracted_json = json_match.group(1).strip()
            else:
                extracted_json = json_match.group(0).strip()
                
            # --- JSON CLEANING AND PARSING ---
            # CRITICAL: Remove non-breaking spaces (\xa0) that models sometimes insert
            cleaned_json = extracted_json.replace(u"\xa0", " ") 
            data = json.loads(cleaned_json)

            # 4. CRITICAL VALIDATION 
            clue = data.get("clue", "").upper()
            number = data.get("number", 0)

            if not clue:
                raise ValueError("Clue word is empty or missing.")
            if clue in all_board_words:
                raise ValueError(f"Clue word '{clue}' is on the board.")
            if not isinstance(number, int) or number < 0 or number > len(team_words):
                raise ValueError(f"Clue number is invalid: {number}.")

            # If all checks pass, return the original data
            return data

        except Exception as e:
            # 5. Non-Crashing Fallback: Ends the turn immediately with no strategic guesses.
            print(f"**CRITICAL FAILURE**: Codemaster failed validation or parsing: {e}. Raw output: {repr(raw)}")
            
            return {"clue": "FAIL_NO_WORD", "number": 0, "reasoning": f"Parsing/Validation Error: {e}"}

    def make_guess(self, 
                   team: str,
                   board: List[str], 
                   clue: str, 
                   number: int,
                   game_state: Dict) -> Dict:
        """Make a guess as the Guesser (JSON output)"""
        if self.role != "guesser":
            raise ValueError("This agent is not initialized as a Guesser")

        available_words = [word for word in board if word not in game_state["guessed_words"]]
        prompt = get_guesser_prompt(team, available_words, clue, number, game_state)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Use configurable max_tokens, default to 300 for guesser
        max_tokens = self.model_config.get("max_tokens", 300)
        raw = self._make_request(messages, max_tokens=max_tokens)

        print("guesser raw output: " + raw)
        available_words_upper = {word.upper() for word in available_words}
        try:
            json_match = None

            # 1. Check 1 (Most likely to succeed): Find content inside ```json{...}```
            json_match = re.search(r"```json\s*(\{.*?})\s*```", raw, re.DOTALL)

            if not json_match:
                # 2. Check 2: Find content inside ```{...}``` (missing 'json' tag)
                json_match = re.search(r"```\s*(\{.*?})\s*```", raw, re.DOTALL)

            if not json_match:
                # 3. Check 3 (Fallback): Find the first raw JSON object, ignoring all surrounding noise
                json_match = re.search(r"(\{.*?})", raw, re.DOTALL)

                if not json_match:
                    raise ValueError(f"No valid JSON code block found in guesser output: {repr(raw)}")

            # --- JSON Extraction ---
            # If groups > 0, we matched a code block (use group(1) for the content)
            # Otherwise, use group(0) for the raw JSON fallback
            if len(json_match.groups()) > 0:
                extracted_json = json_match.group(1).strip()
            else:
                extracted_json = json_match.group(0).strip()

            # CRITICAL: Remove non-breaking spaces (\xa0) that models sometimes insert
            cleaned_json = extracted_json.replace(u"\xa0", " ")
            data = json.loads(cleaned_json)

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from guesser output: {repr(raw)}") from e

        # Validate keys
        if "guess" not in data or "reasoning" not in data:
            raise ValueError(f"Guesser JSON missing required keys: {repr(data)}")
        
        guess = data.get("guess", "").upper()
        
        # Validate that the guess is a real, available word (anti-placeholder/hallucination defense)
        if not guess or guess == "THE WORD YOU CHOOSE": # Specific placeholder defense
            raise ValueError(f"Guesser output contained placeholder text: {guess}")

        if guess not in available_words_upper:
            # Note: We return a fail clue instead of raising to avoid crashing the whole turn
            print(f"**WARNING**: Guesser output '{guess}' is not available on board. Forcing STOP.")
            return {"guess": "STOP", "reasoning": f"Guessed invalid word {guess}. Forcing stop."}
        
        return data
    def receive_guess_feedback(self,
                             guess: str,
                             result: str,
                             clue: str,
                             number: int,
                             game_state: Dict) -> None:
        """Process feedback about a guess"""
        if self.role != 'guesser':
            raise ValueError("This agent is not initialized as a Guesser")

        prompt = GUESSER_FEEDBACK_PROMPT.format(
            guess=guess,
            result=result,
            successful_guesses=game_state.get("current_turn_successes", []),
            unsuccessful_guesses=game_state.get("current_turn_failures", []),
            remaining_guesses=game_state.get("guesses_remaining", 0),
            clue=clue,
            number=number
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Don't need to wait for or process the response
        self._make_request(messages, max_tokens=10)