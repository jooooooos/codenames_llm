# LLM Codenames Benchmark

A framework for benchmarking Large Language Models (LLMs) using the game Codenames. This system tests how well LLMs can work together in a collaborative setting, where a codemaster and guesser from the same model work as a team to identify words on the board.

## Overview

In Codenames, players compete to identify their team's words on the board using one-word clues. Each game features:
- A Codemaster who knows all word assignments and gives clues
- A Guesser who tries to identify team words based on the clue

This benchmark system:
- Supports multiple LLM providers (OpenAI, Gemini, Anthropic, and local HuggingFace models)
- Runs collaborative games where codemaster and guesser work together against the board
- Tracks detailed game metrics (win rate, guess accuracy, turns per game)
- Handles rate limiting and retries
- Provides comprehensive game logs
- Supports local model inference with quantization options

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
HF_TOKEN="your-hugginface-token"
```

## Project Structure

```
codenames_llm/
├── main.py              # Main entry point with model configurations
├── benchmark.py         # Benchmark system and game simulation logic
├── llm_providers.py     # LLM provider implementations (OpenAI, Gemini, Claude, HuggingFace)
├── llm_agent.py         # Agent logic for codemaster/guesser roles
├── prompts.py           # System prompts for different roles
├── game_logger.py       # Game event logging system
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not tracked in git)
├── .gitignore          # Git ignore rules
├── words/
│   └── default.txt      # Codenames word list
└── game_logs/           # Generated game logs and metrics
    └── game_events.log
```

## Usage

### Running the Benchmark

The simplest way to run the benchmark is through [main.py](main.py):

```bash
python main.py
```

By default, this runs 3 collaborative games using the configured model (currently set to Llama 3.1).

### Choosing a Model

Edit the `model_choice` variable in [main.py:66](main.py#L66) to test different models:

```python
model_choice = "llama31"  # Options: "gpt4", "gpt3", "gemini", "claude", "llama31", "qwen3", "mistral"
```

### Available Models

The system currently supports:

**Cloud-based models:**
- `gpt4`: GPT-4 (OpenAI)
- `gpt3`: GPT-3.5-turbo (OpenAI)
- `gemini`: Gemini 2.0 Flash (Google)
- `claude`: Claude 3 Opus (Anthropic)

**Local HuggingFace models:**
- `llama31`: Meta Llama 3.1 8B Instruct
- `qwen3`: Qwen 3 8B (with reasoning mode support)
- `mistral`: Mistral 7B Instruct v0.3

### Game Results

After running, you'll see output like:
```
=== Game Results ===
Model: meta-llama/Llama-3.1-8B-Instruct
Games played: 3
Wins: 2
Win rate: 66.7%
Average turns per game: 8.3
Total correct guesses: 18
Total incorrect guesses: 4
Guess accuracy: 81.8%
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
