# prompts.py

def format_game_history(past_turns: list) -> str:
    """
    Format game history with progressive detail reduction:
    - Recent turns (last 3): Full details with result labels
      Example: "Turn 10: Clue 'OCEAN 2' → WAVE (team word), TIDE (neutral)"

    - Medium-age turns (4-7 back): Abbreviated results using symbols
      Example: "Turn 6: 'SPORT 2' → BALL ✓, FIELD ✓, STADIUM ✗"
      (✓ = team word, ✗ = wrong)

    - Oldest turns (8+ back): Only correct guesses shown
      Example: "T3: 'METAL 2' → GOLD, SILVER"

    This preserves what was guessed and what worked while reducing token count.
    """
    if not past_turns:
        return "No turns played yet"

    history_lines = []
    total_turns = len(past_turns)

    for i, turn in enumerate(past_turns):
        turns_from_end = total_turns - i

        if turns_from_end <= 3:
            # Recent turns: Full details with result labels
            history_lines.append(
                f"Turn {turn['turn_number']}: "
                f"Clue '{turn['clue_word']} {turn['clue_number']}' → "
                f"{', '.join([f'{g} ({r})' for g, r in zip(turn['guesses'], turn['results'])])}"
            )
        elif turns_from_end <= 7:
            # Medium-age turns: Use symbols to compress results
            # ✓ = team word, ✗ = wrong (neutral/opponent/assassin)
            guess_symbols = []
            for guess, result in zip(turn['guesses'], turn['results']):
                symbol = '✓' if result == 'team word' else '✗'
                guess_symbols.append(f"{guess} {symbol}")

            history_lines.append(
                f"Turn {turn['turn_number']}: '{turn['clue_word']} {turn['clue_number']}' → "
                f"{', '.join(guess_symbols)}"
            )
        else:
            # Oldest turns: Only show correct guesses (what actually worked)
            correct_guesses = [g for g, r in zip(turn['guesses'], turn['results'])
                             if r == 'team word']
            if correct_guesses:
                history_lines.append(
                    f"T{turn['turn_number']}: '{turn['clue_word']} {turn['clue_number']}' → "
                    f"{', '.join(correct_guesses)}"
                )
            else:
                # If no correct guesses, still show the clue was given but failed
                history_lines.append(
                    f"T{turn['turn_number']}: '{turn['clue_word']} {turn['clue_number']}' → (none correct)"
                )

    return '\n'.join(history_lines)

CODEMASTER_SYSTEM_PROMPT = """You are playing as the Codemaster in Codenames. Your role is to give clues that help your team guess specific words while avoiding neutral words and the assassin word.

---
**CRITICAL RULES:**
1. Your clue must be a **single word**.
2. **The clue CANNOT be any word currently on the board or any part of them.**
3. The number indicates how many words on the board relate to your clue.
4. You cannot use proper nouns or made-up words.
5. Think strategically about connecting multiple words with a single clue.

---
**OUTPUT FORMAT (MANDATORY JSON CODE BLOCK):**
Your **ENTIRE** response **MUST** be a single JSON object contained within markdown code fences, preceded by the language indicator `json`. **DO NOT** include any other text, commentary, or instructions outside this block.

Example format:
```json
{
  "reasoning": "brief internal reasoning for your choice",
  "clue": "your single word clue that is NOT one of the words on the board",
  "number": number_of_words_related_to_clue
}"""

def get_codemaster_prompt(team: str, team_words: list, neutral_words: list,
                         avoid_words: list, assassin: str, game_state: dict) -> str:
    # Use two-tier history formatting for token efficiency
    guess_history = format_game_history(game_state["past_turns"])

    return f"""You are the Codemaster for {team}. Here is the current game state:

Your team's remaining words to guess: {', '.join(team_words)}
Other words to avoid: {', '.join(avoid_words)}
Neutral words (avoid these): {', '.join(neutral_words)}
Assassin word (CRITICAL to avoid): {assassin}

Game History:
{guess_history}

Your team has {len(team_words)} words left to guess.
**CRITICAL MANDATORY RESPONSE FORMAT:**
    **YOUR ENTIRE RESPONSE MUST BE ONLY A JSON OBJECT INSIDE A ```json ... ``` CODE BLOCK.**
    **The response MUST NOT contain any other text outside the code block.**
    
    Give a strategic clue. DO NOT GENERATE ANY FURTHER TEXT AFTER THE FINAL CLOSING BRACE.
    """
GUESSER_SYSTEM_PROMPT = """You are playing as the Guesser in Codenames. Your role is to interpret the Codemaster's clue and choose words from the board.

Rules:
1. Only guess words that are currently on the board and have not previously been guessed.
2. Consider the clue and number provided.
3. The value for the "guess" field **MUST** be a word currently visible on the board.

**OUTPUT FORMAT (MANDATORY JSON CODE BLOCK):**
Your **ENTIRE** response **MUST** be a single JSON object contained within markdown code fences, preceded by the language indicator `json`. **DO NOT** include any other text, commentary, or instructions outside this block.

Example format:
```json
{
  "reasoning": "reasoning explaining your choice",
  "guess": "the word you choose (must be a word on the board)"
}
"""

def get_guesser_prompt(team: str, board: list, clue: str, number: int, game_state: dict) -> str:
    # Use two-tier history formatting for token efficiency
    guess_history = format_game_history(game_state["past_turns"])

    # Get already guessed words (confirmed role)
    guessed_team_words = [word for turn in game_state["past_turns"] 
                         if turn["team"] == team 
                         for word, result in zip(turn["guesses"], turn["results"])
                         if result == "team word"]
                         
    return f"""You are the Guesser for {team}.

Current Board State:
{', '.join(word for word in board if word not in game_state["guessed_words"])}

Your Codemaster's clue is: {clue} {number}
This means there are {number} words on the board related to '{clue}'

Game History:
{guess_history}

Your team has found these words so far: {', '.join(guessed_team_words)}

Already guessed words this turn: {', '.join(game_state["current_turn_guesses"])}
Remaining guesses for this clue: {game_state["guesses_remaining"]}
**CRITICAL ADHERENCE:** Your response MUST be **ONLY** a JSON object inside a ```json ... ``` code block.
The value for the "guess" field **MUST** be a word currently visible on the board.
"""

GUESSER_FEEDBACK_PROMPT = """After your guess '{guess}', the result was: {result}

Game History:
{game_history}

Previous guesses this turn:
Successful: {successful_guesses}
Unsuccessful: {unsuccessful_guesses}

You have {remaining_guesses} guesses remaining for the clue '{clue} {number}'.
Would you like to make another guess? If so, choose carefully from the remaining board words."""