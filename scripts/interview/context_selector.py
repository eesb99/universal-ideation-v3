"""
Context Selector for Universal Ideation Integration
Provides interactive selection of interview contexts for ideation
"""

from typing import Optional, List, Dict, Callable
from datetime import datetime

from .interview_storage import InterviewStorage, InitiativeStatus
from .models import InterviewContext


class ContextSelector:
    """
    Interactive selector for interview contexts.

    Used by universal-ideation-v3 to query and select interview contexts
    before running ideation.
    """

    def __init__(
        self,
        storage: InterviewStorage = None,
        input_callback: Callable[[str], str] = None,
        output_callback: Callable[[str], None] = None
    ):
        """
        Initialize the context selector.

        Args:
            storage: InterviewStorage instance
            input_callback: Callback for user input (prompt -> response)
            output_callback: Callback for output (message -> None)
        """
        self.storage = storage or InterviewStorage()
        self.input_callback = input_callback or self._default_input
        self.output_callback = output_callback or self._default_output

    def _default_input(self, prompt: str) -> str:
        """Default input using stdin."""
        self.output_callback(prompt)
        try:
            return input().strip()
        except (EOFError, KeyboardInterrupt):
            return ""

    def _default_output(self, message: str):
        """Default output using stdout."""
        print(message)

    def get_available_contexts(
        self,
        domain_hint: str = None,
        status: InitiativeStatus = InitiativeStatus.READY,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get available interview contexts.

        Args:
            domain_hint: Optional domain to filter/sort by similarity
            status: Status filter (default: READY)
            limit: Maximum number to return

        Returns:
            List of context dictionaries sorted by relevance
        """
        if domain_hint:
            # Use semantic similarity to find relevant contexts
            similar = self.storage.find_similar_initiatives(
                domain_hint, threshold=0.5, limit=limit
            )
            contexts = []
            for item in similar:
                init = item['initiative']
                if init['status'] == status.value:
                    init['similarity'] = item['similarity']
                    contexts.append(init)

            # If not enough from similarity, add recent ones
            if len(contexts) < limit:
                all_ready = self.storage.list_initiatives(status=status, limit=limit)
                existing_ids = {c['id'] for c in contexts}
                for init in all_ready:
                    if init['id'] not in existing_ids:
                        init['similarity'] = 0.0
                        contexts.append(init)
                        if len(contexts) >= limit:
                            break

            return contexts
        else:
            # Return all ready contexts sorted by last_updated
            return self.storage.list_initiatives(status=status, limit=limit)

    def select_context_interactive(
        self,
        domain_hint: str = None
    ) -> Optional[InterviewContext]:
        """
        Interactively select a context from available options.

        Args:
            domain_hint: Optional domain to help filter contexts

        Returns:
            Selected InterviewContext or None if cancelled
        """
        contexts = self.get_available_contexts(domain_hint)

        if not contexts:
            self.output_callback("\nNo interview contexts available.")
            self.output_callback("Run /universal-interview first to create context.\n")
            return None

        self.output_callback("\n" + "=" * 60)
        self.output_callback("AVAILABLE INTERVIEW CONTEXTS")
        self.output_callback("=" * 60)

        for i, ctx in enumerate(contexts, 1):
            similarity_badge = ""
            if ctx.get('similarity', 0) > 0.7:
                similarity_badge = " [HIGH MATCH]"
            elif ctx.get('similarity', 0) > 0.5:
                similarity_badge = " [MATCH]"

            self.output_callback(f"\n{i}. {ctx['name']}{similarity_badge}")
            self.output_callback(f"   ID: {ctx['id'][:8]}")
            self.output_callback(f"   Domain: {ctx['original_domain'][:60]}...")
            self.output_callback(f"   Updated: {ctx['last_updated'][:19]}")

            if ctx.get('enriched_domain'):
                self.output_callback(f"   Enriched: {ctx['enriched_domain'][:80]}...")

        self.output_callback("\n" + "-" * 60)
        self.output_callback("Options:")
        self.output_callback("  [1-N] Select context by number")
        self.output_callback("  [s]   Skip - proceed without interview context")
        self.output_callback("  [q]   Quit")
        self.output_callback("")

        choice = self.input_callback("Select context: ")

        if not choice or choice.lower() == 'q':
            return None

        if choice.lower() == 's':
            self.output_callback("Proceeding without interview context.")
            return None

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(contexts):
                selected = contexts[idx]
                context_dict = self.storage.build_context(selected['id'])
                context = InterviewContext.from_dict(context_dict)

                # Mark as active
                self.storage.update_initiative_status(
                    selected['id'],
                    InitiativeStatus.ACTIVE
                )

                self.output_callback(f"\nSelected: {selected['name']}")
                return context
            else:
                self.output_callback("Invalid selection.")
                return None
        except ValueError:
            # Maybe they entered an ID directly
            for ctx in contexts:
                if ctx['id'].startswith(choice):
                    context_dict = self.storage.build_context(ctx['id'])
                    context = InterviewContext.from_dict(context_dict)

                    self.storage.update_initiative_status(
                        ctx['id'],
                        InitiativeStatus.ACTIVE
                    )

                    self.output_callback(f"\nSelected: {ctx['name']}")
                    return context

            self.output_callback("Invalid selection.")
            return None

    def get_context_by_id(self, initiative_id: str) -> Optional[InterviewContext]:
        """
        Get a context directly by ID.

        Args:
            initiative_id: Full or partial initiative ID

        Returns:
            InterviewContext or None if not found
        """
        # Find by partial ID
        initiatives = self.storage.list_initiatives()
        matches = [i for i in initiatives if i['id'].startswith(initiative_id)]

        if not matches:
            return None

        if len(matches) > 1:
            # Return first match (most recent)
            pass

        full_id = matches[0]['id']
        context_dict = self.storage.build_context(full_id)

        if context_dict:
            # Mark as active
            self.storage.update_initiative_status(
                full_id,
                InitiativeStatus.ACTIVE
            )
            return InterviewContext.from_dict(context_dict)

        return None

    def release_context(self, initiative_id: str):
        """
        Release a context after ideation completes.
        Changes status from ACTIVE back to READY.

        Args:
            initiative_id: Initiative ID to release
        """
        initiative = self.storage.get_initiative(initiative_id)
        if initiative and initiative['status'] == InitiativeStatus.ACTIVE.value:
            self.storage.update_initiative_status(
                initiative_id,
                InitiativeStatus.READY
            )

    def format_context_for_prompt(
        self,
        context: InterviewContext,
        include_gaps: bool = True,
        max_length: int = 2000
    ) -> str:
        """
        Format interview context for injection into LLM prompts.

        Args:
            context: The InterviewContext to format
            include_gaps: Whether to include knowledge gaps
            max_length: Maximum character length

        Returns:
            Formatted context string for prompt injection
        """
        return context.get_prompt_context()[:max_length]

    def get_weighted_directives(self, context: InterviewContext) -> Dict[str, float]:
        """
        Extract weighted generation directives from context.

        Currently returns empty dict as user requested keeping default weights.
        This is a hook for future V2 weight adjustment features.

        Args:
            context: The InterviewContext

        Returns:
            Dictionary of dimension -> weight_multiplier (currently empty)
        """
        # User requested to keep default weights, so no modification
        return {}


def create_context_selector(
    input_callback: Callable[[str], str] = None,
    output_callback: Callable[[str], None] = None
) -> ContextSelector:
    """
    Factory function to create a context selector.

    Args:
        input_callback: Optional custom input handler
        output_callback: Optional custom output handler

    Returns:
        Configured ContextSelector instance
    """
    return ContextSelector(
        input_callback=input_callback,
        output_callback=output_callback
    )
