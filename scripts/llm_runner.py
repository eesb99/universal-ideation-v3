"""
Universal Ideation v3.2 - LLM-Integrated Runner with Full Storage

Connects the v3.2 orchestrator to:
- Anthropic Claude API for idea generation and scoring
- Qdrant for vector embeddings and semantic distance
- SQLite for persistent storage
"""

import os
import json
import sys
import uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path.home() / ".env")

# Ensure script directory is in path
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import anthropic
from run_v3 import (
    IdeationOrchestrator,
    OrchestratorConfig,
    SessionPhase
)
from generators.triple_generator import GeneratorMode, ConstraintTemplate
from storage.memory_helper import MemoryHelper
from search.perplexity_search import PerplexitySearch, create_perplexity_search

# Initialize Anthropic client
client = anthropic.Anthropic(
    api_key=os.environ.get("CLAUDE_API_KEY")
)

# Initialize Perplexity search (optional - for web knowledge)
perplexity_client: PerplexitySearch = None
market_context: str = ""

MODEL = "claude-sonnet-4-20250514"

# Global storage helper (initialized in run_full_ideation)
memory_helper = None

# Scoring weights for 8-dimension evaluation (shared across modes)
SCORING_WEIGHTS = {
    "novelty": 0.12, "feasibility": 0.18, "market": 0.18,
    "complexity": 0.12, "scenario": 0.12, "contrarian": 0.10,
    "surprise": 0.10, "cross_domain": 0.08
}


def generate_ideas_batch(domain: str, mode: GeneratorMode, batch_size: int = 10,
                         learnings: list = None, market_context: str = "") -> list:
    """Generate multiple ideas in a single API call for efficiency."""

    mode_descriptions = {
        GeneratorMode.EXPLORER: "EXPLORER using SCAMPER framework for breakthrough novelty",
        GeneratorMode.REFINER: "REFINER using Design Thinking for practical feasibility",
        GeneratorMode.CONTRARIAN: "CONTRARIAN using TRIZ to challenge industry assumptions"
    }

    mode_desc = mode_descriptions.get(mode, mode_descriptions[GeneratorMode.EXPLORER])

    # Add learnings context if available
    learning_context = ""
    if learnings and len(learnings) > 0:
        learning_context = "\n\nPrevious learnings to incorporate:\n"
        for learning in learnings[-5:]:
            learning_context += f"- {learning}\n"

    # Add market context if available
    context_section = ""
    if market_context:
        context_section = f"\n\n{market_context}"

    prompt = f"""You are a {mode_desc}.

Domain: {domain}
{learning_context}{context_section}

Generate {batch_size} DIVERSE and UNIQUE product/service ideas. Each idea should be distinctly different from the others.

Requirements:
- Each idea must solve a different problem or target a different segment
- Vary the innovation types (Breakthrough, Incremental, Disruptive)
- Include a mix of B2B and B2C concepts
- Range from low-tech to high-tech solutions

Return a JSON array with exactly {batch_size} ideas:
[
    {{
        "title": "Product/Service Name",
        "description": "2-3 sentence core concept",
        "target_market": "Specific customer segment",
        "differentiators": ["Key difference 1", "Key difference 2"],
        "price_point": "Estimated price",
        "innovation_type": "Breakthrough/Incremental/Disruptive"
    }},
    ...
]

Return ONLY the JSON array, no other text."""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=max(2048, batch_size * 400),  # Scale with batch size
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text

        # Extract JSON if wrapped in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        ideas = json.loads(content.strip())

        # Ensure it's a list
        if not isinstance(ideas, list):
            ideas = [ideas]

        # Add generator mode to each idea
        for idea in ideas:
            idea["generator_mode"] = mode.value

        return ideas

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return []
    except Exception as e:
        print(f"Batch generation error: {e}")
        return []


def generate_idea_with_llm(domain: str, mode: GeneratorMode, learnings: list = None) -> dict:
    """Generate idea using Claude API based on generator mode."""

    mode_prompts = {
        GeneratorMode.EXPLORER: f"""You are an EXPLORER generating breakthrough ideas.

Domain: {domain}

Use SCAMPER framework to maximize novelty:
- Substitute, Combine, Adapt, Modify, Put to other uses, Eliminate, Reverse

Generate ONE innovative product/service idea. Be creative and push boundaries.

Return JSON only:
{{
    "title": "Product/Service Name",
    "description": "2-3 sentence core concept",
    "target_market": "Specific customer segment",
    "differentiators": ["Key difference 1", "Key difference 2", "Key difference 3"],
    "price_point": "Estimated price",
    "innovation_type": "Breakthrough/Incremental/Disruptive"
}}""",

        GeneratorMode.REFINER: f"""You are a REFINER optimizing for feasibility.

Domain: {domain}

Use Design Thinking to maximize execution potential:
- Empathize with real customer needs
- Define practical problems
- Ideate solutions that can be built

Generate ONE practical, market-ready idea. Focus on execution feasibility.

Return JSON only:
{{
    "title": "Product/Service Name",
    "description": "2-3 sentence core concept",
    "target_market": "Specific customer segment",
    "differentiators": ["Practical advantage 1", "Practical advantage 2"],
    "manufacturing_feasibility": "High/Medium/Low with explanation",
    "regulatory_pathway": "Key compliance requirements",
    "price_point": "Estimated price"
}}""",

        GeneratorMode.CONTRARIAN: f"""You are a CONTRARIAN challenging assumptions.

Domain: {domain}

Use TRIZ to find blue ocean opportunities:
- Identify industry assumptions everyone takes for granted
- Challenge what "must" be true
- Find contradictions to resolve

Generate ONE contrarian idea that breaks industry conventions.

Return JSON only:
{{
    "title": "Product/Service Name",
    "description": "2-3 sentence core concept",
    "target_market": "Underserved segment others ignore",
    "assumption_challenged": "The industry belief this breaks",
    "why_competitors_wont_copy": "Structural barrier",
    "risk_factors": ["Risk 1", "Risk 2"],
    "price_point": "Estimated price"
}}"""
    }

    # Add learnings context if available
    learning_context = ""
    if learnings and len(learnings) > 0:
        learning_context = "\n\nPrevious learnings to incorporate:\n"
        for learning in learnings[-5:]:  # Last 5 learnings
            learning_context += f"- {learning}\n"

    prompt = mode_prompts.get(mode, mode_prompts[GeneratorMode.EXPLORER])
    prompt += learning_context

    # Add market context from Perplexity search if available
    if market_context:
        prompt += f"\n\n{market_context}"

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON from response
        content = response.content[0].text

        # Extract JSON if wrapped in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        idea = json.loads(content.strip())
        idea["generator_mode"] = mode.value
        return idea

    except Exception as e:
        print(f"LLM generation error: {e}")
        return {
            "title": f"Error generating idea",
            "description": str(e),
            "target_market": "N/A",
            "differentiators": [],
            "generator_mode": mode.value
        }


def score_idea_with_llm(idea: dict, domain: str) -> dict:
    """Score idea across 8 dimensions using Claude API."""

    idea_text = json.dumps(idea, indent=2)

    prompt = f"""Score this idea across 8 dimensions (0-100 scale).

Domain: {domain}

Idea:
{idea_text}

Score each dimension and provide brief justification:

1. **Novelty** (12% weight): How statistically rare/unique is this?
2. **Feasibility** (18% weight): Can this be executed with available resources?
3. **Market** (18% weight): Is there demand? Good positioning?
4. **Complexity** (12% weight): Does it leverage network effects/systems thinking?
5. **Scenario** (12% weight): Will this be robust across future scenarios?
6. **Contrarian** (10% weight): Does it challenge assumptions effectively?
7. **Surprise** (10% weight): Does it violate expected schemas in a good way?
8. **Cross-Domain** (8% weight): Does it bridge concepts from other fields?

Return JSON only:
{{
    "novelty": <score>,
    "feasibility": <score>,
    "market": <score>,
    "complexity": <score>,
    "scenario": <score>,
    "contrarian": <score>,
    "surprise": <score>,
    "cross_domain": <score>,
    "justification": {{
        "novelty": "brief reason",
        "feasibility": "brief reason",
        "market": "brief reason",
        "complexity": "brief reason",
        "scenario": "brief reason",
        "contrarian": "brief reason",
        "surprise": "brief reason",
        "cross_domain": "brief reason"
    }}
}}"""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text

        # Extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        scores = json.loads(content.strip())

        # Ensure all dimensions present and return only numeric scores
        dimensions = ["novelty", "feasibility", "market", "complexity",
                      "scenario", "contrarian", "surprise", "cross_domain"]
        result = {}
        for dim in dimensions:
            if dim not in scores:
                result[dim] = 50.0
            else:
                result[dim] = float(scores[dim])

        return result

    except Exception as e:
        print(f"LLM scoring error: {e}")
        return {
            "novelty": 50.0,
            "feasibility": 50.0,
            "market": 50.0,
            "complexity": 50.0,
            "scenario": 50.0,
            "contrarian": 50.0,
            "surprise": 50.0,
            "cross_domain": 50.0
        }


# Note: Embedding generation is handled internally by MemoryHelper.store_idea()


def run_full_ideation(domain: str, iterations: int = 15, minutes: int = 15,
                      threshold: float = 60.0, verbose: bool = True,
                      use_web_search: bool = False):
    """
    Run full v3.2 ideation with LLM integration and full storage.

    Args:
        domain: Domain/focus area for ideation
        iterations: Maximum iterations
        minutes: Maximum runtime in minutes
        threshold: Acceptance score threshold
        verbose: Enable verbose output
        use_web_search: Enable Perplexity web search for market context
    """

    global memory_helper, perplexity_client, market_context

    # Initialize storage
    print("=" * 60)
    print("UNIVERSAL IDEATION v3.2 - FULL LLM + STORAGE MODE")
    print("=" * 60)

    # Initialize MemoryHelper (SQLite + Qdrant + Embeddings)
    print("Initializing storage...")
    memory_helper = MemoryHelper()

    storage_status = []
    if memory_helper.qdrant:
        storage_status.append("Qdrant: Connected")
    else:
        storage_status.append("Qdrant: NOT CONNECTED")

    if memory_helper.embedder:
        storage_status.append("Embeddings: Ready")
    else:
        storage_status.append("Embeddings: NOT AVAILABLE")

    storage_status.append(f"SQLite: {memory_helper.db_path}")

    # Initialize Perplexity search if enabled
    if use_web_search:
        perplexity_client = create_perplexity_search()
        if perplexity_client and perplexity_client.api_key:
            storage_status.append("Perplexity: Connected")
            print("\nGathering market intelligence via web search...")
            try:
                context_results = perplexity_client.get_ideation_context(domain)
                market_context = perplexity_client.format_for_prompt(context_results)
                if market_context:
                    print(f"  Retrieved {len(market_context)} chars of market context")
                    # Show summary of what was found
                    for key, result in context_results.items():
                        status = "OK" if result.success else "FAILED"
                        print(f"    - {key}: {status}")
                else:
                    print("  No market context retrieved")
            except Exception as e:
                print(f"  Web search error: {e}")
                market_context = ""
        else:
            storage_status.append("Perplexity: NOT CONFIGURED (set PERPLEXITY_API_KEY)")
            market_context = ""
    else:
        market_context = ""

    for status in storage_status:
        print(f"  {status}")

    print()
    print(f"Domain: {domain}")
    print(f"Model: {MODEL}")
    print(f"Config: {iterations} iterations, {minutes} min, threshold {threshold}")
    print("=" * 60)
    print()

    # Generate session ID
    session_id = str(uuid.uuid4())[:8]

    # Create orchestrator with config
    config = OrchestratorConfig(
        acceptance_threshold=threshold,
        enable_verification=True,
        enable_reflection=True,
        enable_atomic_novelty=True,
        atomic_novelty_weight=0.3,
        min_novelty_score=40.0
    )

    orchestrator = IdeationOrchestrator(domain=domain, config=config)

    # Track ideas for storage
    ideas_stored = 0

    # Callback for iteration progress
    def on_iteration(iteration: int, result):
        nonlocal ideas_stored

        status = "[+]" if result.accepted else "[-]"
        title = result.idea.get("title", "Unknown")[:40]
        mode = result.idea.get("generator_mode", "?")
        print(f"{status} {iteration:3d}: {title} | {result.final_score:.1f} | {mode}")

        if verbose and result.accepted:
            print(f"        Description: {result.idea.get('description', 'N/A')[:60]}...")

        # Store accepted ideas in database
        if result.accepted and memory_helper:
            try:
                # store_idea handles both SQLite and Qdrant embedding storage
                memory_helper.store_idea(
                    session_id=session_id,
                    domain=domain,
                    mode=mode,
                    concept_name=result.idea.get("title", "Untitled"),
                    description=result.idea.get("description", ""),
                    target_audience=result.idea.get("target_market", ""),
                    differentiation=json.dumps(result.idea.get("differentiators", [])),
                    scores=result.dimension_scores,
                    weighted_score=result.final_score
                )
                ideas_stored += 1
            except Exception as e:
                print(f"        Storage error: {e}")

    # Run with LLM callbacks
    results = orchestrator.run(
        max_iterations=iterations,
        max_minutes=minutes,
        idea_generator=generate_idea_with_llm,
        score_evaluator=score_idea_with_llm,
        on_iteration=on_iteration
    )

    # Print summary
    print()
    print("=" * 60)
    print("SESSION COMPLETE")
    print("=" * 60)
    print(f"Total iterations: {results.total_iterations}")
    print(f"Accepted ideas:   {len(results.accepted_ideas)}")
    print(f"Rejected ideas:   {len(results.rejected_ideas)}")
    print(f"Average score:    {results.average_score:.1f}")
    print(f"Best score:       {results.best_score:.1f}")
    print(f"Escape attempts:  {results.escape_attempts}")
    print()
    print("STORAGE:")
    print(f"  Ideas stored in SQLite: {ideas_stored}")
    print(f"  Database: {memory_helper.db_path}")
    if memory_helper.qdrant:
        try:
            collection_info = memory_helper.qdrant.get_collection(memory_helper.collection_name)
            print(f"  Qdrant vectors: {collection_info.points_count}")
        except:
            print("  Qdrant vectors: N/A")
    print()

    # Print top ideas
    if results.accepted_ideas:
        print("TOP IDEAS:")
        print("-" * 60)
        sorted_ideas = sorted(results.accepted_ideas,
                              key=lambda x: x.final_score, reverse=True)
        for i, result in enumerate(sorted_ideas[:5], 1):
            print(f"\n{i}. {result.idea.get('title', 'Unknown')} (Score: {result.final_score:.1f})")
            print(f"   {result.idea.get('description', 'N/A')}")
            if "differentiators" in result.idea:
                print(f"   Key: {', '.join(result.idea['differentiators'][:2])}")

    print()

    # Export to JSON
    output_dir = Path(_SCRIPT_DIR).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"ideation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    export_data = {
        "session_id": session_id,
        "domain": domain,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "iterations": iterations,
            "minutes": minutes,
            "threshold": threshold,
            "enable_verification": config.enable_verification,
            "enable_reflection": config.enable_reflection,
            "enable_atomic_novelty": config.enable_atomic_novelty,
            "atomic_novelty_weight": config.atomic_novelty_weight,
            "min_novelty_score": config.min_novelty_score
        },
        "results": {
            "total_iterations": results.total_iterations,
            "accepted_count": len(results.accepted_ideas),
            "rejected_count": len(results.rejected_ideas),
            "average_score": results.average_score,
            "best_score": results.best_score,
            "duration_seconds": results.duration_seconds
        },
        "v3_2_features": {
            "escape_attempts": results.escape_attempts,
            "escape_successful": results.escape_successful,
            "plateau_info": results.plateau_info,
            "generator_mode_distribution": results.generator_mode_distribution,
            "dimension_averages": results.dimension_averages,
            "learnings": results.learnings
        },
        "ideas": [
            {
                "title": r.idea.get("title"),
                "description": r.idea.get("description"),
                "score": r.final_score,
                "mode": r.generator_mode.value,
                "dimensions": r.dimension_scores,
                "atomic_novelty": {
                    "final_score": r.atomic_novelty_result.final_score,
                    "tier": r.atomic_novelty_result.novelty_tier.value if hasattr(r.atomic_novelty_result.novelty_tier, 'value') else str(r.atomic_novelty_result.novelty_tier),
                    "confidence": r.atomic_novelty_result.confidence
                } if r.atomic_novelty_result else None
            }
            for r in results.accepted_ideas
        ]
    }

    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"Results exported to: {output_file}")

    return results


def run_batch_ideation(domain: str, target_ideas: int = 100, batch_size: int = 10,
                       threshold: float = 60.0, verbose: bool = True,
                       use_web_search: bool = False):
    """
    Run batch ideation - generates multiple ideas per API call for efficiency.

    Args:
        domain: Domain/focus area for ideation
        target_ideas: Target number of ideas to generate (default 100)
        batch_size: Ideas per API call (default 10)
        threshold: Acceptance score threshold
        verbose: Enable verbose output
        use_web_search: Enable Perplexity web search for market context
    """

    global memory_helper, perplexity_client, market_context

    print("=" * 60)
    print("UNIVERSAL IDEATION v3.2 - BATCH MODE")
    print("=" * 60)

    # Initialize storage
    print("Initializing storage...")
    memory_helper = MemoryHelper()

    storage_status = []
    if memory_helper.qdrant:
        storage_status.append("Qdrant: Connected")
    else:
        storage_status.append("Qdrant: NOT CONNECTED")

    if memory_helper.embedder:
        storage_status.append("Embeddings: Ready")
    else:
        storage_status.append("Embeddings: NOT AVAILABLE")

    storage_status.append(f"SQLite: {memory_helper.db_path}")

    # Initialize Perplexity search if enabled
    if use_web_search:
        perplexity_client = create_perplexity_search()
        if perplexity_client and perplexity_client.api_key:
            storage_status.append("Perplexity: Connected")
            print("\nGathering market intelligence via web search...")
            try:
                context_results = perplexity_client.get_ideation_context(domain)
                market_context = perplexity_client.format_for_prompt(context_results)
                if market_context:
                    print(f"  Retrieved {len(market_context)} chars of market context")
                    for key, result in context_results.items():
                        status = "OK" if result.success else "FAILED"
                        print(f"    - {key}: {status}")
            except Exception as e:
                print(f"  Web search error: {e}")
                market_context = ""
        else:
            storage_status.append("Perplexity: NOT CONFIGURED")
            market_context = ""
    else:
        market_context = ""

    for status in storage_status:
        print(f"  {status}")

    print()
    print(f"Domain: {domain}")
    print(f"Model: {MODEL}")
    print(f"Target: {target_ideas} ideas | Batch size: {batch_size} | Threshold: {threshold}")
    print("=" * 60)
    print()

    session_id = str(uuid.uuid4())[:8]
    start_time = datetime.now()

    # Tracking
    all_ideas = []
    accepted_ideas = []
    rejected_ideas = []
    ideas_stored = 0

    # Calculate batches needed
    num_batches = (target_ideas + batch_size - 1) // batch_size
    modes = [GeneratorMode.EXPLORER, GeneratorMode.REFINER, GeneratorMode.CONTRARIAN]

    print(f"Generating {num_batches} batches across {len(modes)} modes...")
    print()

    batch_num = 0
    for round_num in range((num_batches + len(modes) - 1) // len(modes)):
        for mode in modes:
            if len(all_ideas) >= target_ideas:
                break

            batch_num += 1
            print(f"[Batch {batch_num}/{num_batches}] Mode: {mode.value}...")

            # Generate batch
            ideas = generate_ideas_batch(
                domain=domain,
                mode=mode,
                batch_size=batch_size,
                learnings=[r["title"] for r in accepted_ideas[-10:]] if accepted_ideas else None,
                market_context=market_context
            )

            if not ideas:
                print(f"  Warning: No ideas generated")
                continue

            print(f"  Generated {len(ideas)} ideas, scoring...")

            # Track batch-specific counts
            batch_accepted = 0
            batch_rejected = 0

            # Score each idea
            for i, idea in enumerate(ideas):
                scores = score_idea_with_llm(idea, domain)

                # Calculate weighted score using shared constant
                weighted_score = sum(scores.get(dim, 50) * w for dim, w in SCORING_WEIGHTS.items())

                idea["scores"] = scores
                idea["weighted_score"] = weighted_score
                all_ideas.append(idea)

                # Accept/reject based on threshold
                if weighted_score >= threshold:
                    accepted_ideas.append(idea)
                    batch_accepted += 1
                    status = "[+]"

                    # Store in database
                    if memory_helper:
                        try:
                            memory_helper.store_idea(
                                session_id=session_id,
                                domain=domain,
                                mode=idea.get("generator_mode", "unknown"),
                                concept_name=idea.get("title", "Untitled"),
                                description=idea.get("description", ""),
                                target_audience=idea.get("target_market", ""),
                                differentiation=json.dumps(idea.get("differentiators", [])),
                                scores=scores,
                                weighted_score=weighted_score
                            )
                            ideas_stored += 1
                        except Exception as e:
                            print(f"    Storage error: {e}")
                else:
                    rejected_ideas.append(idea)
                    batch_rejected += 1
                    status = "[-]"

                if verbose:
                    title = idea.get("title", "Unknown")[:35]
                    print(f"    {status} {title} | {weighted_score:.1f}")

            print(f"  Batch complete: +{batch_accepted} accepted, +{batch_rejected} rejected (total: {len(accepted_ideas)})")
            print()

        if len(all_ideas) >= target_ideas:
            break

    duration = (datetime.now() - start_time).total_seconds()

    # Print summary
    print("=" * 60)
    print("BATCH SESSION COMPLETE")
    print("=" * 60)
    print(f"Total ideas generated: {len(all_ideas)}")
    print(f"Accepted ideas:        {len(accepted_ideas)}")
    print(f"Rejected ideas:        {len(rejected_ideas)}")
    if all_ideas:
        print(f"Acceptance rate:       {len(accepted_ideas)/len(all_ideas)*100:.1f}%")
    else:
        print("Acceptance rate:       N/A")
    print(f"Duration:              {duration:.1f}s ({duration/60:.1f} min)")
    print(f"Ideas/minute:          {len(all_ideas)/(duration/60):.1f}" if duration > 0 else "N/A")
    print()

    if accepted_ideas:
        avg_score = sum(i["weighted_score"] for i in accepted_ideas) / len(accepted_ideas)
        best_score = max(i["weighted_score"] for i in accepted_ideas)
        print(f"Average score:         {avg_score:.1f}")
        print(f"Best score:            {best_score:.1f}")

    print()
    print("STORAGE:")
    print(f"  Ideas stored in SQLite: {ideas_stored}")
    print(f"  Database: {memory_helper.db_path}")
    if memory_helper.qdrant:
        try:
            collection_info = memory_helper.qdrant.get_collection(memory_helper.collection_name)
            print(f"  Qdrant vectors: {collection_info.points_count}")
        except:
            print("  Qdrant vectors: N/A")
    print()

    # Print top ideas
    if accepted_ideas:
        print("TOP 10 IDEAS:")
        print("-" * 60)
        sorted_ideas = sorted(accepted_ideas, key=lambda x: x["weighted_score"], reverse=True)
        for i, idea in enumerate(sorted_ideas[:10], 1):
            print(f"\n{i}. {idea.get('title', 'Unknown')} (Score: {idea['weighted_score']:.1f})")
            print(f"   {idea.get('description', 'N/A')[:100]}...")

    print()

    # Export to JSON
    output_dir = Path(_SCRIPT_DIR).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"batch_ideation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    export_data = {
        "session_id": session_id,
        "domain": domain,
        "timestamp": datetime.now().isoformat(),
        "mode": "batch",
        "config": {
            "target_ideas": target_ideas,
            "batch_size": batch_size,
            "threshold": threshold,
            "use_web_search": use_web_search
        },
        "results": {
            "total_generated": len(all_ideas),
            "accepted_count": len(accepted_ideas),
            "rejected_count": len(rejected_ideas),
            "acceptance_rate": len(accepted_ideas)/len(all_ideas) if all_ideas else 0,
            "duration_seconds": duration
        },
        "ideas": sorted(accepted_ideas, key=lambda x: x["weighted_score"], reverse=True)
    }

    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"Results exported to: {output_file}")

    return accepted_ideas


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal Ideation v3.2 - Full LLM + Storage Mode")
    parser.add_argument("domain", help="Domain/focus area for ideation")
    parser.add_argument("-i", "--iterations", type=int, default=15, help="Max iterations (standard mode)")
    parser.add_argument("-m", "--minutes", type=int, default=15, help="Max minutes (standard mode)")
    parser.add_argument("-t", "--threshold", type=float, default=60.0, help="Acceptance threshold")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-w", "--web-search", action="store_true",
                        help="Enable Perplexity web search for market context (requires PERPLEXITY_API_KEY)")

    # Batch mode options
    parser.add_argument("-b", "--batch", action="store_true",
                        help="Enable batch mode (generate multiple ideas per API call)")
    parser.add_argument("-n", "--target", type=int, default=100,
                        help="Target number of ideas (batch mode, default 100)")
    parser.add_argument("-s", "--batch-size", type=int, default=10,
                        help="Ideas per API call (batch mode, default 10)")

    args = parser.parse_args()

    if args.batch:
        # Batch mode
        run_batch_ideation(
            domain=args.domain,
            target_ideas=args.target,
            batch_size=args.batch_size,
            threshold=args.threshold,
            verbose=args.verbose,
            use_web_search=args.web_search
        )
    else:
        # Standard mode
        run_full_ideation(
            domain=args.domain,
            iterations=args.iterations,
            minutes=args.minutes,
            threshold=args.threshold,
            verbose=args.verbose,
            use_web_search=args.web_search
        )
