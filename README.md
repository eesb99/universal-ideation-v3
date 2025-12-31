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
- **Universal Interview** - Adaptive domain interview for enriched ideation context (NEW)

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

## Universal Interview (NEW)

Enrich your ideation with deep domain context through an adaptive interview process.

### How It Works

1. Run `/universal-interview "your domain"` first
2. Answer questions about problem space, constraints, intent, etc.
3. System builds a rich context profile
4. Run `/universal-ideation-v3` and select your interview context
5. Ideas are generated with full awareness of your situation

### Interview Skill Commands

```bash
# Start new interview
/universal-interview "sustainable packaging"

# Resume or list in-progress interviews
/universal-interview --continue

# List all initiatives
/universal-interview --list

# Export context as Markdown
/universal-interview --export [initiative-id]

# Delete an initiative
/universal-interview --delete [initiative-id]
```

### CLI Usage

```bash
cd ~/.claude/skills/universal-ideation-v3

# Start interactive interview
python3 scripts/interview_runner.py "sustainable packaging"

# List initiatives
python3 scripts/interview_runner.py --list

# Export context
python3 scripts/interview_runner.py --export abc123

# View stats
python3 scripts/interview_runner.py --stats
```

### Interview Dimensions

The interview explores 7 key dimensions:

| Dimension | Purpose |
|-----------|---------|
| **Problem Space** | What pain points exist? Who suffers? |
| **Constraints** | Budget, timeline, regulations, geography |
| **Assumptions** | Hidden beliefs that might be wrong |
| **Intent** | Strategic goal - disrupt, defend, optimize? |
| **Preferences** | What excites vs. bores you |
| **Existing Solutions** | Competitors, prior attempts |
| **Resources** | Assets, capabilities, relationships |

### Using Context in Ideation

Run ideation with interview context:

```bash
# Interactive context selection
python3 scripts/llm_runner.py "sustainable packaging" -c -v

# Use specific context ID
python3 scripts/llm_runner.py "sustainable packaging" --context-id abc123 -v
```

### Template Scaffolds

The interview can apply pre-built constraint templates:

| Template | Applied When |
|----------|--------------|
| **BOOTSTRAP** | Budget-constrained startup |
| **ENTERPRISE** | Corporate context |
| **REGULATED** | Healthcare, finance, etc. |
| **SUSTAINABLE** | Environmental focus |

## Options

### llm_runner.py (Full Mode)

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --iterations` | 15 | Max iterations |
| `-m, --minutes` | 15 | Max duration |
| `-t, --threshold` | 60.0 | Acceptance score |
| `-v, --verbose` | false | Show progress |
| `-w, --web-search` | false | Enable Perplexity web search for market context |
| `-c, --context` | false | Enable interactive interview context selection |
| `--context-id` | - | Use specific interview context ID |

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
- Location: `data/ideation.db`
- Tables: ideas, sessions, learnings
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
├── SKILL.md              # Ideation skill definition
├── INTERVIEW_SKILL.md    # Interview skill definition (NEW)
├── README.md             # This file
├── requirements.txt      # Dependencies
├── scripts/
│   ├── run_v3.py        # Main orchestrator (stub mode)
│   ├── llm_runner.py    # LLM-integrated runner (full mode)
│   ├── interview_runner.py  # Interview CLI runner (NEW)
│   ├── backup.py        # Database backup tool
│   ├── generators/      # Triple Generator
│   ├── gates/           # Quality gates
│   ├── evaluators/      # Cognitive diversity
│   ├── learning/        # DARLING + reflection
│   ├── escape/          # Plateau escape
│   ├── novelty/         # Atomic novelty
│   ├── search/          # Web search (Perplexity)
│   ├── storage/         # Persistence (SQLite + Qdrant)
│   └── interview/       # Interview module (NEW)
│       ├── interview_storage.py   # Interview persistence
│       ├── interview_engine.py    # Core interview logic
│       ├── models.py              # Data models
│       └── context_selector.py    # Context selection for ideation
├── tests/               # Unit tests
├── data/                # Runtime SQLite
├── backups/             # Database backups
└── output/              # Generated results
```

## License

MIT

## Version

v3.2 (2025-12-29)
