---
name: universal-interview
description: Adaptive domain interview that builds rich context for ideation through fluid, curiosity-driven conversation
argument-hint: "[domain] | --continue | --list | --export [id] | --delete [id]"
allowed-tools: Task, Bash, Read, Write
complexity: "moderate"
updated: "2025-12-31"
author: "Universal Interview v1.0"
version: "1.0"
---

# Universal Interview v1.0

Standalone skill that conducts adaptive, fluid interviews to build rich domain context before ideation.

## Quick Start

```bash
# Start new interview
/universal-interview "sustainable packaging"

# Resume or list past interviews
/universal-interview --continue

# List all initiatives
/universal-interview --list

# Export context as Markdown
/universal-interview --export [initiative-id]

# Delete an initiative
/universal-interview --delete [initiative-id]
```

## Purpose

Enriches the thin domain string input with deep context before feeding into `universal-ideation-v3`:

**Before:** `"sustainable packaging"` (7 characters of context)

**After:** Rich structured context covering:
- Problem space clarity
- Constraints discovered
- Hidden assumptions surfaced
- Strategic intent understood
- Preferences captured
- Existing solutions mapped
- Resources inventoried

## Architecture

### Standalone + Database-Linked

```
┌─────────────────────┐         ┌──────────────────────┐
│ /universal-interview │ ──────▶ │  SQLite + Qdrant     │
│   (this skill)       │  store  │  (shared database)   │
└─────────────────────┘         └──────────────────────┘
                                         │
                                         │ query
                                         ▼
                                ┌──────────────────────┐
                                │ /universal-ideation  │
                                │  (ideation skill)    │
                                └──────────────────────┘
```

### 7 Interview Dimensions

| Dimension | Purpose | Example Questions |
|-----------|---------|-------------------|
| **Problem Space** | What pain points exist? Who suffers? | "What frustration sparked this exploration?" |
| **Constraints** | Budget, timeline, regulations, geography | "What boundaries are you working within?" |
| **Assumptions** | Hidden beliefs that might be wrong | "What do you take for granted here?" |
| **Intent** | Strategic goal - disrupt, defend, optimize? | "What does success look like in 2 years?" |
| **Preferences** | What excites vs. bores the user | "What kind of ideas energize you?" |
| **Existing Solutions** | Competitors, prior attempts | "What's already been tried?" |
| **Resources** | Assets, capabilities, relationships | "What unfair advantages do you have?" |

### Interview Behavior

- **Tone:** Curious collaborator ("I'm curious - who do you imagine using this?")
- **Questions:** Open-ended, follow user's energy
- **Progress:** Explicit indication ("We've covered constraints, exploring resources next...")
- **Completion:** Adaptive sensing detects diminishing returns
- **Knowledge:** Injects domain insights with cited sources + counter-perspectives
- **Scaffolding:** Starts minimal, escalates based on inferred expertise

### Context Output

Interview produces:

```json
{
  "initiative_id": "uuid",
  "initiative_name": "Sustainable Packaging Malaysia",
  "status": "ready",
  "created_at": "2025-12-31T10:00:00Z",
  "last_updated": "2025-12-31T10:30:00Z",

  "enriched_domain": "Sustainable packaging solutions for Malaysian F&B SMEs...",

  "dimensions": {
    "problem_space": {
      "response": "...",
      "confidence": "high",
      "source": "user"
    },
    "constraints": {
      "response": "Budget under RM200k, 6-month timeline...",
      "confidence": "medium",
      "source": "user"
    }
  },

  "gaps_flagged": ["regulatory_landscape", "competitor_pricing"],
  "template_scaffold": "BOOTSTRAP",
  "source_attributions": {
    "user": ["problem_space", "constraints", "preferences"],
    "injected": ["market_trends"]
  }
}
```

### Status Lifecycle

| Status | Meaning |
|--------|---------|
| **draft** | Interview in progress, not complete |
| **ready** | Interview complete, available for ideation |
| **active** | Currently being used by ideation run |
| **archived** | Old, kept for reference |

### Persistence

**SQLite Tables:**
- `initiatives` - Core initiative metadata
- `interview_sessions` - Individual interview sessions
- `interview_responses` - Q&A pairs with confidence
- `interview_analytics` - Metrics for learning loop (V2)

**Qdrant Collection:**
- `interview_contexts` - Semantic vectors for similarity matching

### Continuity Features

- **Semantic similarity** detects related past initiatives
- **Resume** interrupted sessions exactly where left off
- **Live binding** - context updates affect generation in ideation
- **User-configurable** retention period

## Sub-Commands

### New Interview
```bash
/universal-interview "domain description"
```
Starts fresh interview. If similar initiative detected, asks to continue or start new.

### Continue
```bash
/universal-interview --continue
```
Lists in-progress interviews, allows selection to resume.

### List
```bash
/universal-interview --list
```
Shows all initiatives with status, created date, last updated.

### Export
```bash
/universal-interview --export [initiative-id]
```
Outputs Markdown summary of context for documentation/sharing.

### Delete
```bash
/universal-interview --delete [initiative-id]
```
Removes initiative and all associated data (with confirmation).

## Integration with Ideation

After interview completes:

1. Run `/universal-ideation-v3`
2. System shows list of available contexts (interactive selection)
3. User picks context
4. Ideation uses enriched domain + structured context
5. High-scoring ideas feed back to improve interview patterns (V2)

## Edge Case Handling

| Scenario | Behavior |
|----------|----------|
| **Rambling user** | Summarize and confirm, gently redirect |
| **Adversarial input** | Push back gently, detect if unusable |
| **"Just generate"** | Offer quick-mode (3 more questions) or skip |
| **Mid-interview pivot** | Ask explicitly: merge or restart? |
| **Session interrupted** | Resume exactly where left off |
| **API failure** | Graceful degradation, note the gap |
| **Web data contradicts user** | Present finding, explore discrepancy |

## Template Scaffolding

Interview uses constraint templates as starting scaffolds:

| Template | Applied When | Constraints Pre-filled |
|----------|--------------|------------------------|
| **BOOTSTRAP** | Budget-constrained startup | <$50k, MVP focus, 3mo timeline |
| **ENTERPRISE** | Corporate context | Scalability, compliance, integration |
| **REGULATED** | Healthcare, finance, etc. | Regulatory pathway, safety-first |
| **SUSTAINABLE** | Environmental focus | Circular economy, ethical sourcing |

## MVP Scope (v1.0)

**Included:**
- Core interview flow with 7 dimensions
- Adaptive sensing for completion
- SQLite + Qdrant persistence
- Semantic similarity for initiative matching
- Context schema + enriched domain output
- Markdown export
- All sub-commands
- Status lifecycle
- Template scaffolding
- Unit tests

**V2 Backlog:**
- Perplexity web search during interview
- Change-detection re-validation
- Learning loop from ideation results
- Cross-session trend analytics
- Counter-perspective offering

## Instructions for Claude

When this skill is invoked:

1. **Parse arguments** - detect sub-command (new, continue, list, export, delete)
2. **Check for similar initiatives** via semantic similarity (for new interviews)
3. **Run interview engine**:
   ```bash
   cd ~/.claude/skills/universal-ideation-v3
   python3 scripts/interview_runner.py "domain" [--flags]
   ```
4. **Store context** in SQLite + Qdrant
5. **Show synthesized context** for user confirmation
6. **Handle corrections** inline before finalizing

### Interview Flow

```python
# Pseudocode for interview engine
1. Initialize or resume session
2. Load template scaffold if applicable
3. For each dimension (following user energy):
   a. Ask open-ended question
   b. Listen and follow up based on response
   c. Detect engagement level
   d. Inject knowledge if gaps detected (cite sources)
   e. Track confidence per response
4. Detect diminishing returns (response length, vagueness, fatigue signals)
5. Synthesize context
6. Show summary for confirmation
7. Accept inline corrections
8. Store final context with status="ready"
```

## Testing

```bash
cd ~/.claude/skills/universal-ideation-v3
python -m pytest tests/test_interview*.py -v
```

## Version History

- **v1.0** (2025-12-31): Initial release with core interview flow, 7 dimensions, adaptive sensing, persistence, export
