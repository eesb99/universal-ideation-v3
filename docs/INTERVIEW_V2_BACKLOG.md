# Universal Interview V2 Backlog

Features planned for future versions of the Universal Interview module.

## V2 Features (Planned)

### 1. Perplexity Web Search During Interview

**Priority:** High

Integrate real-time web search to enrich interview responses with market intelligence.

**Behavior:**
- When user mentions a domain or industry, fetch current trends
- Present findings with citations during interview
- Offer counter-perspectives to challenge assumptions
- Flag discrepancies between user beliefs and market data

**Implementation:**
- Use existing Perplexity search module
- Inject search context into follow-up question generation
- Store source attributions with web-sourced data

---

### 2. Change-Detection Re-validation

**Priority:** High

Automatically detect when stored context may be stale based on market changes.

**Behavior:**
- Periodically check web for changes relevant to stored initiatives
- When significant changes detected, prompt user for re-validation
- Delta-focused: "I noticed X has changed since we last talked..."
- Explore changes to understand user's updated perspective

**Implementation:**
- Background job to check initiatives older than N days
- Semantic comparison of new search results vs. stored context
- Re-validation prompts integrated into interview flow

---

### 3. Learning Loop from Ideation Results

**Priority:** High

Use ideation outcomes to improve future interviews.

**Behavior:**
- Track which interview dimensions correlated with high-scoring ideas
- Statistical correlation: "Interviews mentioning X produced 15% higher scores"
- Apply learnings to future interviews:
  - Question prioritization (ask high-impact dimensions first)
  - Probing depth (dig deeper on predictive topics)
  - Knowledge injection (surface patterns from successful contexts)

**Implementation:**
- `interview_analytics` table (already created) to store metrics
- Correlation analysis between interview responses and idea scores
- Pattern extraction similar to existing ReflectEvo module

---

### 4. Cross-Session Trend Analytics

**Priority:** Medium

Track how user understanding and context evolves over multiple sessions.

**Behavior:**
- Visualize how constraints, intent, assumptions change over time
- Identify patterns in user exploration (widening vs. narrowing focus)
- Detect when user is circling vs. making progress
- Surface insights: "You've shifted from X focus to Y over last 3 sessions"

**Implementation:**
- Time-series analysis of dimension responses
- Semantic drift detection using embeddings
- Dashboard or report generation

---

### 5. Counter-Perspective Offering

**Priority:** Medium

Proactively challenge user assumptions with opposing viewpoints.

**Behavior:**
- After user states assumption, offer: "Though some argue the opposite..."
- Use web search to find contrarian viewpoints
- Balance validation with healthy skepticism
- Track when counter-perspectives led to better ideas

**Implementation:**
- LLM-generated counter-perspectives
- Web search for opposing expert opinions
- A/B testing to measure impact on idea quality

---

### 6. Multi-User Collaboration

**Priority:** Low (Future)

Allow multiple stakeholders to contribute to an interview context.

**Behavior:**
- Invite collaborators to add perspectives
- Reconcile conflicting viewpoints
- Track attribution per contributor
- Aggregate confidence across contributors

**Implementation:**
- User/collaborator model in database
- Merge conflict resolution logic
- Per-response contributor tracking

---

### 7. Interview Templates Library

**Priority:** Low (Future)

Pre-built interview templates for specific domains or use cases.

**Templates to Consider:**
- **SaaS Product Discovery** - Focus on user personas, pricing, integration
- **Hardware Innovation** - Manufacturing, supply chain, regulations
- **Service Design** - Customer journey, touchpoints, operational complexity
- **Market Entry** - Geographic, competitive, regulatory landscape
- **Pivot Exploration** - Existing assets, capabilities, constraints

**Implementation:**
- Template definitions with pre-filled dimensions
- Question customization per template
- Template suggestion based on domain analysis

---

### 8. Voice Interview Mode

**Priority:** Low (Future)

Support voice-based interview for more natural conversation.

**Behavior:**
- Speech-to-text for user responses
- Text-to-speech for interviewer questions
- Natural conversation flow without typing
- Transcription stored for context

**Implementation:**
- Integration with speech APIs
- Real-time transcription
- Audio file storage and playback

---

## Technical Debt (To Address)

1. **Embedding model configurability** - Allow different embedding models
2. **Session timeout handling** - More graceful handling of long pauses
3. **Offline mode** - Work without Qdrant for simpler deployments
4. **Performance optimization** - Batch embedding operations
5. **Export format options** - JSON, YAML in addition to Markdown

---

## Metrics to Track (V2)

| Metric | Purpose |
|--------|---------|
| Interview completion rate | % of started interviews that complete |
| Questions per session | Engagement depth |
| Time to completion | User experience |
| Context usage rate | % of ideation runs using interview context |
| Quality correlation | Relationship between interview depth and idea scores |
| Re-validation trigger rate | How often stale contexts are detected |
| Learning application rate | How often learned patterns are applied |

---

## Dependencies for V2

- Perplexity API access (for web search features)
- Background job infrastructure (for change detection)
- Analytics visualization (for trend analysis)
- Extended storage for analytics data

---

*Last Updated: 2025-12-31*
