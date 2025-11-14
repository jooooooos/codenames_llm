# LLM Codenames Benchmark

A framework for benchmarking Large Language Models (LLMs) against each other using the game Codenames. This system allows different LLMs to compete in 2v2 matches, where each team consists of a codemaster and a guesser from the same model.

## Overview

In Codenames, players compete to identify their team's words on the board using one-word clues. Each team has:
- A Codemaster who knows all word assignments and gives clues
- A Guesser who tries to identify team words based on the clue

This benchmark system:
- Supports multiple LLM providers (OpenAI, Gemini, Anthropic)
- Tracks detailed game metrics
- Handles rate limiting and retries
- Provides comprehensive game logs

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-codenames-benchmark
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
ANTHROPIC_API_KEY=your-anthropic-key
```

## Project Structure

```
codenames_benchmark/
├── __init__.py
├── llm_providers.py    # LLM provider implementations
├── llm_agent.py       # Agent logic for codemaster/guesser
├── benchmark.py       # Main benchmark system
├── prompts.py        # Prompts for different roles
└── words/           
    └── default.txt   # Codenames word list
```

## Usage

1. Run a benchmark match:
```python
from benchmark import CodeNamesBenchmark

# Configure models
model_configs = {
    "gpt4": {
        "type": "openai",
        "model_name": "gpt-4",
        "api_key": "your-api-key",
        "temperature": 0.7
    },
    "gemini": {
        "type": "gemini",
        "model_name": "gemini-pro",
        "api_key": "your-api-key",
        "temperature": 0.7
    }
}

# Initialize and run benchmark
benchmark = CodeNamesBenchmark(log_dir="game_logs")
metrics = benchmark.run_matchup(
    team_a_config=model_configs["gpt4"],
    team_b_config=model_configs["gemini"],
    num_games=3
)
```

2. View results:
```python
for team, stats in metrics.items():
    print(f"\n{team.upper()}:")
    print(f"Wins: {stats['wins']}/{stats['games_played']}")
    print(f"Win Rate: {stats['win_rate']:.2%}")
    print(f"Correct guesses: {stats['total_correct_guesses']}")
    print(f"Incorrect guesses: {stats['total_incorrect_guesses']}")
    print(f"Average words per clue: {stats['average_words_per_clue']:.2f}")
```

## Adding New Models

To add support for a new LLM provider:

1. Create a new class in `llm_providers.py` that inherits from `BaseLLM`:
```python
class NewProviderLLM(BaseLLM):
    def __init__(self, config: Dict):
        super().__init__(config)
        # Initialize provider-specific client
        
    def generate(self, messages: List[Dict], max_tokens: int) -> str:
        # Implement provider-specific generation logic
        pass
```

2. Add the provider to the factory function:
```python
def create_llm(config: Dict) -> BaseLLM:
    llm_type = config['type'].lower()
    if llm_type == 'new_provider':
        return NewProviderLLM(config)
    # ... other providers
```

## Metrics Tracked

- Win rate
- Correct/incorrect guesses
- Average words per clue
- Game duration
- Turn-by-turn statistics
- Clue effectiveness

## Game Logs

Detailed game logs are saved in the specified log directory:
- `game_events.log`: Turn-by-turn game events
- `game_{id}.json`: Detailed game data
- `benchmark_metrics.json`: Aggregate statistics

## Contributing

Contributions are welcome! Areas for improvement:
- Additional LLM providers
- Enhanced metrics and analysis
- UI improvements
- Custom word lists
- Tournament system

## License

MIT

## Acknowledgments

Based on the game Codenames by Vlaada Chvátil.
