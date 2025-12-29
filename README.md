# Universal Ideation v3.2

Science-grounded autonomous ideation system with self-improving mechanisms.

## Features

- **Triple Generator** - Explorer, Refiner, Contrarian modes
- **8-Dimension Scoring** - Novelty, Feasibility, Market, Complexity, Scenario, Contrarian, Surprise, Cross-Domain
- **DARLING Learning** - Diversity-aware reward calculation
- **Atomic Novelty (NovAScore)** - 0.94 accuracy claim-level novelty detection
- **Verification Gates** - Quality checkpoints
- **Reflection Learning** - Self-improving pattern extraction
- **Plateau Escape** - Avoid local optima

## Installation

### As Claude Code Skill

1. Copy folder to `~/.claude/skills/universal-ideation-v3/`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Invoke with:
   ```bash
   /universal-ideation-v3 "your domain"
   ```

### Optional: Vector Search

For semantic similarity features:
```bash
pip install qdrant-client sentence-transformers
docker run -p 6333:6333 qdrant/qdrant
```

## Quick Start

### Via Skill Command
```bash
/universal-ideation-v3 "e-commerce innovation"
/universal-ideation-v3 "sustainable opportunity in Malaysia"
/universal-ideation-v3 "ai start up innovation"
/universal-ideation-v3 "sustainable packaging innovation"
```

### Via CLI
```bash
cd ~/.claude/skills/universal-ideation-v3
python3 scripts/run_v3.py "your domain" --verbose
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --iterations` | 30 | Max iterations |
| `-m, --minutes` | 30 | Max duration |
| `-t, --threshold` | 65.0 | Acceptance score |
| `-v, --verbose` | false | Show progress |
| `--test` | false | Stub mode for testing |

## Output

Results saved to:
- `output/ideation_YYYYMMDD_HHMMSS.json`
- `data/ideation.db` (SQLite)

## Testing

```bash
python -m pytest tests/ -v
```

## Structure

```
universal-ideation-v3/
├── SKILL.md              # Skill definition
├── README.md             # This file
├── requirements.txt      # Dependencies
├── scripts/
│   ├── run_v3.py        # Main orchestrator
│   ├── generators/      # Triple Generator
│   ├── gates/           # Quality gates
│   ├── evaluators/      # Cognitive diversity
│   ├── learning/        # DARLING + reflection
│   ├── escape/          # Plateau escape
│   ├── novelty/         # Atomic novelty
│   └── storage/         # Persistence
├── tests/               # 74 unit tests
├── data/                # Runtime SQLite
└── output/              # Generated results
```

## License

MIT

## Version

v3.2 (2025-12-29)
