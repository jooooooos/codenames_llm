# prompts.py

CODEMASTER_SYSTEM_PROMPT = """You are playing as the Codemaster in Codenames. Your role is to give clues that help your team guess specific words while avoiding opponent's words, neutral words, and the assassin word.

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
  "clue": "your single word clue that is NOT one of the words on the board",
  "number": number_of_words_related_to_clue,
  "reasoning": "brief internal reasoning for your choice "
}"""

def get_codemaster_prompt(team: str, team_words: list, neutral_words: list, 
                         opponent_words: list, assassin: str, game_state: dict) -> str:
    # Format past guesses into a readable history
    guess_history = []
    for turn in game_state["past_turns"]:
        guess_history.append(
            f"Turn {turn['turn_number']} - {turn['team']}: "
            f"Clue '{turn['clue_word']} {turn['clue_number']}' → "
            f"Guesses: {', '.join([f'{g} ({r})' for g, r in zip(turn['guesses'], turn['results'])])}"
        )

    return f"""You are the Codemaster for {team}. Here is the current game state:

Your team's remaining words to guess: {', '.join(team_words)}
Opponent's words (avoid these): {', '.join(opponent_words)}
Neutral words (avoid these): {', '.join(neutral_words)}
Assassin word (CRITICAL to avoid): {assassin}

Game History:
{chr(10).join(guess_history) if guess_history else "No turns played yet"}

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
  "guess": "the word you choose (must be a word on the board)",
  "reasoning": "reasoning explaining your choice"
}
"""

def get_guesser_prompt(team: str, board: list, clue: str, number: int, game_state: dict) -> str:
    # Format past guesses into a readable history
    guess_history = []
    for turn in game_state["past_turns"]:
        guess_history.append(
            f"Turn {turn['turn_number']} - {turn['team']}: "
            f"Clue '{turn['clue_word']} {turn['clue_number']}' → "
            f"Guesses: {', '.join([f'{g} ({r})' for g, r in zip(turn['guesses'], turn['results'])])}"
        )
    
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
{chr(10).join(guess_history) if guess_history else "No turns played yet"}

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