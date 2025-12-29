---
name: universal-ideation-v3
description: Science-grounded autonomous ideation with Triple Generator, DARLING learning, 8-dimension scoring, and NovAScore atomic novelty
argument-hint: "[domain/focus area]"
allowed-tools: Task, Bash, Read, Write
complexity: "complex"
updated: "2025-12-29"
author: "Universal Ideation v3.2"
version: "3.2"
---

# Universal Ideation v3.2

Science-grounded autonomous ideation system with self-improving mechanisms.

## Quick Start

```bash
/universal-ideation-v3 "e-commerce innovation"
/universal-ideation-v3 "sustainable opportunity in Malaysia"
/universal-ideation-v3 "ai start up innovation"
/universal-ideation-v3 "sustainable packaging innovation"
```

## Installation

1. Copy this folder to `~/.claude/skills/universal-ideation-v3/`
2. Install dependencies: `pip install -r requirements.txt`
3. (Optional) Start Qdrant: `docker-compose up -d` in skill folder

## v3.2 Architecture

### Core Innovations (Research-Backed)

| Component | Purpose | Science Base |
|-----------|---------|--------------|
| **Triple Generator** | Explorer, Refiner, Contrarian modes | Dual-pathway creativity model |
| **Semantic Distance Gate** | Reject too-similar ideas | Prevents mode collapse |
| **DARLING Reward** | Quality + Diversity + Exploration | Diversity-aware RL (2024) |
| **Cognitive Diversity** | 4 evaluator personas + debate | Cognitive heterogeneity research |
| **8-Dimension Scoring** | +Surprise +Cross-Domain | CANU framework + conceptual blending |
| **Plateau Escape** | Don't stop at local optima | Curiosity-driven exploration |
| **Atomic Novelty** | NovAScore claim-level analysis | 0.94 accuracy vs 0.83 cosine |
| **Verification Gates** | Mandatory checkpoints | Structural quality enforcement |
| **Reflection Learning** | Real-time pattern extraction | Self-improving generation |

### 8-Dimension Weights

| Dimension | Weight | Measures |
|-----------|--------|----------|
| Novelty | 12% | Statistical rarity |
| Feasibility | 18% | Execution capability |
| Market | 18% | Demand + positioning |
| Complexity | 12% | Network effects |
| Scenario | 12% | Future robustness |
| Contrarian | 10% | Assumption challenging |
| Surprise | 10% | Schema violation |
| Cross-Domain | 8% | Analogical distance |

### NovAScore: Atomic Novelty (v3.2)

Claims-based novelty detection achieving 0.94 accuracy vs 0.83 for cosine similarity.

**Pipeline:**
```
Idea -> ACU Decompose -> NLI Detection -> Salience Weighting -> Hybrid Score
```

| Component | Function | Output |
|-----------|----------|--------|
| **ACU Decomposer** | Break idea into atomic claims | 5-15 claims per idea |
| **NLI Detector** | Entailment/Contradiction/Neutral | Novelty per claim |
| **Salience Weighter** | Weight by importance | Critical to Low |
| **Hybrid Scorer** | Combine signals | 0-100 NovAScore |

**Novelty Tiers:**

| Tier | Score | Meaning |
|------|-------|---------|
| Breakthrough | 90+ | Fundamentally new |
| Highly Novel | 75-89 | Significant innovation |
| Novel | 60-74 | Meaningful novelty |
| Incremental | 40-59 | Small improvement |
| Derivative | <40 | Mostly restates prior art |

## Execution

When invoked, this skill:

1. **Initializes** Qdrant + SQLite storage + Atomic Novelty scorer
2. **Runs** Triple Generator with adaptive mode selection
3. **Evaluates** via 4 cognitive diversity personas
4. **Scores** 8 dimensions with DARLING rewards
5. **Assesses** Atomic Novelty via NovAScore (ACU -> NLI -> Salience)
6. **Learns** via watcher pattern extraction
7. **Escapes** plateaus with divergent generation
8. **Exports** to database + JSON

### CLI Interface

```bash
cd ~/.claude/skills/universal-ideation-v3
python3 scripts/run_v3.py "domain" --iterations 30 --minutes 30 --verbose
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations, -i` | 30 | Max iterations |
| `--minutes, -m` | 30 | Max duration |
| `--threshold, -t` | 65.0 | Acceptance score |
| `--output, -o` | auto | Output JSON path |
| `--verbose, -v` | false | Show each iteration |
| `--test` | false | Run with stub generators |
| `--no-atomic-novelty` | false | Disable NovAScore analysis |
| `--atomic-novelty-weight` | 0.3 | Weight for atomic novelty in scoring |
| `--llm-novelty` | false | Use LLM for enhanced decomposition |
| `--min-novelty` | 40.0 | Minimum novelty score threshold |

## Prerequisites

**Required:**
- Python 3.8+
- numpy

**Optional (for vector search):**
```bash
pip install qdrant-client sentence-transformers
docker-compose up -d  # Start Qdrant
```

## Database

**SQLite:** `data/ideation.db`
- ideas, sessions, learnings, formulations tables

**Qdrant:** `localhost:6333` (optional)
- universal_ideas collection (384-dim embeddings)

## Output

Results saved to:
- `output/ideation_YYYYMMDD_HHMMSS.json`
- SQLite database for persistence
- Qdrant for semantic search (if enabled)

## Checkpoints

| Interval | Agent | Purpose |
|----------|-------|---------|
| Every 5 | contrarian-disruptor | Challenge assumptions |
| Every 10 | first-principles-analyst | Strategic validation |
| Every 15 | Cross-domain injection | Force analogies |

## Instructions for Claude

When this skill is invoked:

1. **Parse domain** from arguments (default: "general innovation")
2. **Check dependencies** installed (numpy required, qdrant optional)
3. **Run orchestrator** via CLI:
   ```bash
   cd ~/.claude/skills/universal-ideation-v3
   python3 scripts/run_v3.py "domain" -v
   ```
4. **For quick ideation** (no full session): Use Task tool with triple generator pattern:
   - Launch 3 parallel agents: explorer, refiner, contrarian modes
   - Score with 8 dimensions
   - Return top ideas
5. **Save results** to database via MemoryHelper

### Quick Ideation Pattern (Direct Task)

```python
# Launch 3 generator modes in parallel
Task("Explorer mode", prompt=f"Generate {domain} idea maximizing novelty...", subagent_type="creative-ideation-specialist")
Task("Refiner mode", prompt=f"Optimize {domain} idea for feasibility...", subagent_type="creative-ideation-specialist")
Task("Contrarian mode", prompt=f"Challenge {domain} assumptions...", subagent_type="contrarian-disruptor")

# Score each with 8 dimensions
# Return ranked results
```

### Full Session Pattern

```bash
cd ~/.claude/skills/universal-ideation-v3
python3 scripts/run_v3.py "dairy-free protein" --iterations 30 --verbose
```

## Testing

Run included tests:
```bash
cd ~/.claude/skills/universal-ideation-v3
python -m pytest tests/ -v
```

## Version History

- **v3.2** (2025-12-28): Atomic Novelty (NovAScore) with ACU decomposition, NLI detection, salience weighting
- **v3.1** (2025-12-27): Verification Gates, Reflection Learning, Dynamic Cross-Domain
- **v3.0** (2025-12-27): Science-grounded rewrite with DARLING, Triple Generator, 8 dimensions
- **v2.0** (2025-12-26): Added contrarian checkpoints, scenario planning
- **v1.0** (2025-12-25): Initial 7-agent system with continuous learning
