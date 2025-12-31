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
- **Web Search Integration** - Real-time market intelligence via Perplexity API
- **Interview Context Integration** - Enriched ideation using universal-interview skill
- **Batch Mode** - Generate multiple ideas per API call (5x faster)

## Architecture

![Universal Ideation v3.2 Architecture](docs/architecture.png)

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

### Via CLI (Stub Mode)
```bash
cd ~/.claude/skills/universal-ideation-v3
python3 scripts/run_v3.py "your domain" --verbose
```

### Via LLM Runner (Full Mode with Claude API)

Requires Anthropic API key in `~/.env`:
```bash
CLAUDE_API_KEY=sk-ant-xxxxx
```

Optional: Add Perplexity API key for web search:
```bash
PERPLEXITY_API_KEY=pplx-xxxxx
```

Run with full LLM integration + storage:
```bash
cd ~/.claude/skills/universal-ideation-v3
python3 scripts/llm_runner.py "your domain" -i 30 -m 30 -v
```

This mode:
- Uses Claude API for idea generation and scoring
- Stores ideas in SQLite database
- Stores embeddings in Qdrant for semantic search
- Exports full v3.2 statistics (DARLING learnings, atomic novelty, etc.)

### With Web Search (Market Intelligence)

Enable real-time market context from Perplexity:
```bash
python3 scripts/llm_runner.py "protein beverages" -i 10 -m 5 -w -v
```

This fetches:
- **Domain trends** - Current market developments
- **Market gaps** - Underserved opportunities
- **Emerging tech** - Relevant technologies
- **Consumer insights** - Behavior patterns

The market context is injected into idea generation prompts for more grounded, trend-aware ideas.

### With Interview Context (Deep Domain Understanding)

Enrich ideation with structured interview context from the universal-interview skill:

```bash
# First, run an interview to build context
cd ~/.claude/skills/universal-interview
python3 scripts/interview_runner.py "sustainable protein beverages"

# Then use the context in ideation
cd ~/.claude/skills/universal-ideation-v3
python3 scripts/llm_runner.py "protein beverages" --context-id [ID] -b -n 10 -v

# Or use interactive context selection
python3 scripts/llm_runner.py "protein beverages" -c -b -n 10 -v
```

The interview context provides:
- **Problem Space** - Pain points and user needs
- **Constraints** - Budget, timeline, regulations
- **Assumptions** - Hidden beliefs to challenge
- **Intent** - Strategic goals
- **Preferences** - What excites vs. bores
- **Existing Solutions** - Competitors and gaps
- **Resources** - Assets and capabilities

Both skills share the database at `~/.claude/data/ideation.db`.

### Batch Mode (5x Faster)

Generate multiple ideas per API call:

```bash
# Generate 100 ideas in batch mode
python3 scripts/llm_runner.py "your domain" -b -n 100 -s 10 -v

# With web search
python3 scripts/llm_runner.py "your domain" -b -n 100 -w -v

# With interview context
python3 scripts/llm_runner.py "your domain" -b -n 100 --context-id [ID] -v
```

| Mode | Ideas/min | 100 ideas |
|------|-----------|-----------|
| Standard | ~1 | ~100 min |
| Batch | ~5 | ~20 min |

## Options

### llm_runner.py (Full Mode)

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --iterations` | 15 | Max iterations (standard mode) |
| `-m, --minutes` | 15 | Max duration (standard mode) |
| `-t, --threshold` | 60.0 | Acceptance score |
| `-v, --verbose` | false | Show progress |
| `-w, --web-search` | false | Enable Perplexity web search |
| `-b, --batch` | false | Enable batch mode |
| `-n, --target` | 100 | Target ideas (batch mode) |
| `-s, --batch-size` | 10 | Ideas per API call (batch mode) |
| `-c, --context` | false | Interactive interview context selection |
| `--context-id` | - | Specific interview context ID |

### run_v3.py (Stub Mode)

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --iterations` | 30 | Max iterations |
| `-m, --minutes` | 30 | Max duration |
| `-t, --threshold` | 65.0 | Acceptance score |
| `-v, --verbose` | false | Show progress |
| `--test` | false | Stub mode for testing |

## Storage

### SQLite Database
- Location: `~/.claude/data/ideation.db` (shared with universal-interview)
- Tables: ideas, sessions, learnings, initiatives, interview_responses
- Persists all accepted ideas with scores

### Qdrant Vector Database (Optional)
- Location: `localhost:6333`
- Collection: `universal_ideas` (384-dim embeddings)
- Enables semantic similarity search

Start Qdrant:
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

## Output

Results saved to:
- `output/ideation_YYYYMMDD_HHMMSS.json` - Full session export
- `data/ideation.db` - SQLite persistence
- Qdrant vectors - Semantic embeddings (if enabled)

## Backup

Backup your ideas database (SQLite + Qdrant vectors):

```bash
cd ~/.claude/skills/universal-ideation-v3

# Full backup (SQLite + Qdrant + JSON)
python3 scripts/backup.py backup

# Named backup
python3 scripts/backup.py backup -n "my_backup"

# SQLite only (skip Qdrant)
python3 scripts/backup.py backup --no-qdrant

# View statistics
python3 scripts/backup.py stats

# List all backups
python3 scripts/backup.py list

# Export all ideas to JSON
python3 scripts/backup.py export -o my_ideas.json

# Restore full backup (SQLite + Qdrant)
python3 scripts/backup.py restore backup_file.db

# Restore SQLite only
python3 scripts/backup.py restore backup_file.db --no-qdrant
```

### Backup Files

| File | Contents |
|------|----------|
| `*_backup.db` | SQLite database (ideas, sessions, learnings) |
| `*_qdrant.snapshot` | Qdrant vector embeddings |
| `*_backup.json` | Full JSON export |

Backups saved to `backups/` folder.

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
│   ├── run_v3.py        # Main orchestrator (stub mode)
│   ├── llm_runner.py    # LLM-integrated runner (full mode)
│   ├── backup.py        # Database backup tool
│   ├── generators/      # Triple Generator
│   ├── gates/           # Quality gates
│   ├── evaluators/      # Cognitive diversity
│   ├── learning/        # DARLING + reflection
│   ├── escape/          # Plateau escape
│   ├── novelty/         # Atomic novelty
│   ├── search/          # Web search (Perplexity)
│   └── storage/         # Persistence (SQLite + Qdrant)
├── tests/               # 74 unit tests
├── data/                # Runtime SQLite
├── backups/             # Database backups
└── output/              # Generated results
```

## License

MIT

## Version

v3.3 (2025-12-31) - Added batch mode, interview context integration, shared database
