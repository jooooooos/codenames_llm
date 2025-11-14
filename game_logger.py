from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import json
import time
from pathlib import Path
import logging

@dataclass
class TurnLog:
    turn_number: int
    team: str
    model_name: str
    clue_word: str
    clue_number: int
    guesses: List[str]
    correct_guesses: List[str]
    remaining_team_words: List[str]
    time_taken: float

@dataclass
class GameLog:
    game_id: int
    team_a_model: str
    team_b_model: str
    start_time: float
    initial_board: List[str]
    team_a_words: List[str]
    team_b_words: List[str]
    neutral_words: List[str]
    assassin: str
    turns: List[TurnLog]
    winner: Optional[str] = None
    end_time: Optional[float] = None
    winning_reason: Optional[str] = None

class GameLogger:
    def __init__(self, log_dir: str = "game_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'game_events.log'),
                logging.StreamHandler()  # Also print to console
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Current game state
        self.current_game: Optional[GameLog] = None
        
    def start_game(self, 
                  game_id: int,
                  team_a_model: str,
                  team_b_model: str,
                  initial_board: List[str],
                  team_a_words: List[str],
                  team_b_words: List[str],
                  neutral_words: List[str],
                  assassin: str):
        """Start logging a new game"""
        self.current_game = GameLog(
            game_id=game_id,
            team_a_model=team_a_model,
            team_b_model=team_b_model,
            start_time=time.time(),
            initial_board=initial_board,
            team_a_words=team_a_words.copy(),
            team_b_words=team_b_words.copy(),
            neutral_words=neutral_words.copy(),
            assassin=assassin,
            turns=[]
        )
        
        self.logger.info(f"Starting game {game_id}: {team_a_model} vs {team_b_model}")
        self.logger.info(f"Initial board: {', '.join(initial_board)}")
        self.logger.info(f"Team A words: {', '.join(team_a_words)}")
        self.logger.info(f"Team B words: {', '.join(team_b_words)}")
        self.logger.info(f"Assassin word: {assassin}")

    def log_turn(self,
                turn_number: int,
                team: str,
                model_name: str,
                clue_word: str,
                clue_number: int,
                guesses: List[str],
                correct_guesses: List[str],
                remaining_team_words: List[str]):
        """Log details about a single turn"""
        if not self.current_game:
            raise ValueError("No game in progress")
            
        turn = TurnLog(
            turn_number=turn_number,
            team=team,
            model_name=model_name,
            clue_word=clue_word,
            clue_number=clue_number,
            guesses=guesses,
            correct_guesses=correct_guesses,
            remaining_team_words=remaining_team_words,
            time_taken=time.time()
        )
        
        self.current_game.turns.append(turn)
        
        self.logger.info(f"Turn {turn_number} - {team} ({model_name}):")
        self.logger.info(f"Clue given: {clue_word} {clue_number}")
        self.logger.info(f"Guesses made: {', '.join(guesses)}")
        self.logger.info(f"Correct guesses: {', '.join(correct_guesses)}")
        self.logger.info(f"Remaining team words: {', '.join(remaining_team_words)}")

    def end_game(self, winner: Optional[str], winning_reason: str):
        """End the current game and save its log"""
        if not self.current_game:
            raise ValueError("No game in progress")
            
        self.current_game.winner = winner
        self.current_game.end_time = time.time()
        self.current_game.winning_reason = winning_reason
        
        self.logger.info(f"Game {self.current_game.game_id} ended")
        self.logger.info(f"Winner: {winner}")
        self.logger.info(f"Reason: {winning_reason}")
        
        # Save detailed game log as JSON
        game_log_path = self.log_dir / f"game_{self.current_game.game_id}.json"
        with open(game_log_path, 'w') as f:
            json.dump(asdict(self.current_game), f, indent=2)
        
        # Clear current game
        self.current_game = None

    def get_game_summary(self, game_id: int) -> Dict:
        """Load and summarize a specific game's log"""
        game_log_path = self.log_dir / f"game_{game_id}.json"
        if not game_log_path.exists():
            raise ValueError(f"No log found for game {game_id}")
            
        with open(game_log_path, 'r') as f:
            game_data = json.load(f)
            
        total_turns = len(game_data['turns'])
        
        return {
            'game_id': game_id,
            'team_a_model': game_data['team_a_model'],
            'team_b_model': game_data['team_b_model'],
            'winner': game_data['winner'],
            'total_turns': total_turns,
            'game_duration': game_data['end_time'] - game_data['start_time'],
            'clues_given': [
                (turn['clue_word'], turn['clue_number']) 
                for turn in game_data['turns']
            ],
            'guesses_made': sum(len(turn['guesses']) for turn in game_data['turns']),
            'correct_guesses': sum(len(turn['correct_guesses']) for turn in game_data['turns']),
            'winning_reason': game_data['winning_reason']
        }