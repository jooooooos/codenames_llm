# Codenames LLM Benchmark - Project Overview

## Purpose

This project is an **LLM Codenames Benchmark System** that evaluates how Large Language Models collaborate in the word-guessing game Codenames. It tests LLM team performance where two agents (Codemaster and Guesser) work together, with support for **persona-based agent behavior** to study how personality characteristics affect agent performance and inter-agent communication.

## What is Codenames?

Codenames is a cooperative word-guessing game where:
- A **Codemaster** provides one-word clues with a number indicating how many words relate to that clue
- A **Guesser** attempts to identify the correct words on a 25-word board
- The team wins by finding all 9 team words while avoiding neutral words and a single assassin word

## Key Research Questions

This codebase helps answer:
- How do personas affect LLM agent communication and game performance?
- Does persona-role alignment matter (e.g., analytical linguist as codemaster)?
- How does partner awareness (persona sharing) affect collaboration?
- What's the baseline (no persona) vs. persona-enhanced performance?
- Can persona-based agents simulate diverse human communication styles?

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Entry Points                          │
│  main.py (single game) | run_full_experiment.py (441 combos) │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Model & LLM Configuration                        │
│  model_configs[choice] → create_llm(config) → LLM instance   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│          Agent Initialization & Persona Injection             │
│  LLMAgent(role='codemaster'/'guesser') +                     │
│  initialize_role(persona_id, partner_persona, shared_persona)│
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│            CodeNamesBenchmark.simulate_game_collab()         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. Generate board (9 team + 8 neutral + 1 assassin) │   │
│  │ 2. Loop turns (max 20):                             │   │
│  │    a) Codemaster.give_clue() → JSON {clue, number}  │   │
│  │    b) Guesser.make_guess() loop → JSON {guess}      │   │
│  │    c) Validate guess (team/neutral/assassin/invalid)│   │
│  │    d) Log turn data with metadata                   │   │
│  │ 3. Track metrics & game state                       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  Results & Logging                           │
│  GameLogger.log_turn() → game_logs/{exp_id}_{cm}_{g}.json    │
│  Aggregate metrics → benchmark_metrics.json                  │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

| File | Lines | Purpose |
|------|-------|---------|
| [main.py](main.py) | 163 | Entry point for running individual games with configurable models and personas |
| [benchmark.py](benchmark.py) | 343 | Core game simulation logic implementing Codenames rules and turn management |
| [llm_agent.py](llm_agent.py) | 266 | Agent abstraction for Codemaster and Guesser roles with persona injection |
| [llm_providers.py](llm_providers.py) | 298 | LLM backend abstraction supporting multiple providers (OpenAI, Gemini, Claude, local) |
| [prompts.py](prompts.py) | 163 | System prompts and game state formatting for agent instructions |
| [persona_loader.py](persona_loader.py) | 92 | Utility for loading and caching the 20 diverse personas from JSON |
| [game_logger.py](game_logger.py) | 157 | Logging infrastructure for capturing turn-by-turn game events |
| [run_full_experiment.py](run_full_experiment.py) | 291 | Executes full experiment across all persona combinations (up to 441) |
| [personas.json](personas.json) | - | Configuration file with 20 diverse persona profiles |
| [default.txt](default.txt) | - | Codenames word list for generating game boards |
| [CodenamesLLM.ipynb](CodenamesLLM.ipynb) | - | Interactive Jupyter notebook for running experiments |
| [RunFullExperiment.ipynb](RunFullExperiment.ipynb) | - | Full persona experiment orchestration with progress tracking |

## Game Flow

### 1. Board Generation
- Creates a 25-word board from the word list
- Splits into: 9 team words, 8 neutral words, 7 opponent words, 1 assassin word
- Only the team's 9 words need to be found to win

### 2. Turn Loop (max 20 turns)

#### Codemaster Phase
1. Analyzes remaining team words
2. Provides a single-word clue and a number (how many words relate)
3. Output format: JSON `{"clue": "word", "number": N}`

#### Guesser Phase
1. Receives the clue
2. Makes up to N+1 guesses sequentially
3. Output format: JSON `{"guess": "word"}` or `{"guess": "PASS"}`
4. Can stop early if uncertain

#### Validation
- **Team word (correct)**: Continue guessing (up to limit)
- **Neutral word**: End turn immediately
- **Assassin word**: Game over, team loses
- **Invalid word**: Forced stop, end turn

### 3. Game End Conditions
- **Win**: All 9 team words guessed
- **Lose**: Assassin word guessed
- **Timeout**: Max turns (20) exceeded

## Persona System

### Available Personas

The system includes 20 diverse personas covering various:
- **Demographics**: Age, background, culture
- **Professions**: Linguist, comedian, teacher, poker player, engineer, artist, etc.
- **Cognitive styles**: Analytical vs. intuitive
- **Communication styles**: Verbose vs. concise
- **Risk tolerance**: Conservative vs. bold

Example personas:
- Dr. Sarah Kim (analytical linguist)
- Marcus Thompson (quick-witted comedian)
- Elena Rodriguez (patient teacher)
- David Chen (strategic poker player)

### Experimental Conditions

The system supports 4 conditions for studying persona effects:

1. **No personas** (baseline): Both agents use default behavior
2. **Codemaster only**: Codemaster has persona, guesser is default
3. **Guesser only**: Guesser has persona, codemaster is default
4. **Both agents**: Both have personas (same or different)

### Persona Sharing

When enabled, agents know their partner's background and can adapt communication accordingly. This tests whether mutual awareness improves collaboration.

## Supported LLM Providers

| Provider | Models | Configuration |
|----------|--------|---------------|
| **OpenAI** | GPT-4, GPT-3.5-turbo | API key via environment |
| **Google** | Gemini 2.0 Flash | API key via environment |
| **Anthropic** | Claude 3 Opus | API key via environment |
| **Local (HuggingFace)** | Llama 3.1 8B, Qwen3 8B, Mistral 7B, Gemma 2 9B | Local inference via vLLM |

## Data Collection

### Turn-by-Turn Logs

Each game generates detailed logs in `game_logs/` directory:

```json
{
  "turn": 1,
  "role": "codemaster",
  "clue": "animal",
  "number": 3,
  "guesses": ["dog", "cat", "bird"],
  "results": ["correct", "correct", "pass"],
  "generation_time": 1.23,
  "reasoning": "...",
  "metadata": {...}
}
```

### Metrics Tracked

- **Performance**: Win rate, turns to completion, guess accuracy
- **Efficiency**: Clues per word, early termination rate
- **Timing**: Agent response latency (generation times)
- **Collaboration**: Guess patterns, communication alignment
- **Errors**: Invalid outputs, JSON parsing failures

## Running Experiments

### Single Game

```bash
python main.py
```

Interactive prompts will guide you through:
1. Model selection
2. Persona configuration (codemaster/guesser)
3. Persona sharing settings

### Full Experiment

```bash
python run_full_experiment.py
```

Executes all combinations:
- 21 codemaster personas (including no-persona baseline)
- 21 guesser personas (including no-persona baseline)
- Total: 441 combinations

Configurable via `max_persona_id` parameter to run smaller subsets:
- `max_persona_id = 5`: 36 combinations (5x5 + baseline)
- `max_persona_id = 10`: 121 combinations
- `max_persona_id = 20`: 441 combinations (full experiment)

### Jupyter Notebooks

- [CodenamesLLM.ipynb](CodenamesLLM.ipynb): Interactive experimentation
- [RunFullExperiment.ipynb](RunFullExperiment.ipynb): Full experiment with progress tracking

## Advanced Features

### Progressive History Reduction

Game history is formatted with different detail levels to manage token context:
- **Recent turns**: Full details with clues and guesses
- **Medium-age turns**: Symbol-based summaries (✓/✗)
- **Oldest turns**: High-level summaries

### Crash Resistance

Experiments can resume from intermediate results if interrupted.

### JSON Validation

Robust parsing with multiple fallback strategies:
1. Standard JSON parsing
2. Regex extraction from markdown code blocks
3. Fuzzy matching for malformed outputs

### Rate Limiting

Built-in retry logic with exponential backoff for API failures.

## Dependencies

### Core Libraries

```
transformers      # Hugging Face model loading
torch             # PyTorch for local inference
vllm              # Fast inference engine
bitsandbytes      # 4-bit quantization support
python-dotenv     # Environment variables
huggingface_hub   # Model repository access
notebook          # Jupyter support
```

### API Requirements

Set environment variables for cloud providers:
```bash
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

## Research Context

This codebase implements ideas from NeurIPS 2024 research on persona-based LLM agents. It enables systematic study of:

1. **Role-persona alignment**: Does matching persona traits to role requirements improve performance?
2. **Communication adaptation**: How do personas affect clue-giving and guessing strategies?
3. **Collaboration dynamics**: Does knowing partner's persona improve team coordination?
4. **Baseline comparison**: Quantifying the impact of persona vs. generic agents

## Output

### Game Logs

Individual game files: `game_logs/{experiment_id}_cm{X}_g{Y}_game{N}.json`

Contains:
- Complete turn history
- Agent reasoning traces
- Performance metrics
- Board state and outcomes

### Aggregate Metrics

`benchmark_metrics.json` contains:
- Per-combination statistics
- Win rates and completion times
- Efficiency metrics
- Error rates

## Recent Development

Recent commits show focus on:
- Persona system implementation and refinement
- Full experiment loop with scalable persona combinations
- Model switching to Gemma 2 9B for local inference
- Mutual persona knowledge (persona sharing feature)
- Reducing experiment length via `max_persona_id` parameter

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up API keys (if using cloud models)
4. Run a single game: `python main.py`
5. Or run full experiment: `python run_full_experiment.py`
6. Analyze results in `game_logs/` directory

## Future Extensions

Potential areas for expansion:
- Additional persona dimensions (cultural, linguistic)
- More sophisticated collaboration metrics
- Multi-game learning and adaptation
- Opponent team simulation (competitive mode)
- Real-time human-agent collaboration experiments
