#!/usr/bin/env python3
"""
Interview Runner for Universal Interview Skill
CLI interface with sub-commands for interview management
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List, Callable
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from interview import (
    InterviewStorage,
    InterviewEngine,
    InterviewContext,
    InterviewState,
    InterviewDimension,
    InitiativeStatus,
    TemplateScaffold,
    Question
)


class InterviewRunner:
    """
    CLI runner for the Universal Interview skill.

    Supports sub-commands:
    - new: Start a new interview
    - continue: Resume or list in-progress interviews
    - list: Show all initiatives
    - export: Export context as Markdown
    - delete: Remove an initiative
    """

    def __init__(
        self,
        storage: InterviewStorage = None,
        llm_callback: Callable[[str], str] = None,
        input_callback: Callable[[str], str] = None,
        output_callback: Callable[[str], None] = None
    ):
        """
        Initialize the runner.

        Args:
            storage: Storage instance (creates default if None)
            llm_callback: Callback for LLM queries (prompt -> response)
            input_callback: Callback for user input (prompt -> response)
            output_callback: Callback for output (message -> None)
        """
        self.storage = storage or InterviewStorage()
        self.engine = InterviewEngine(storage=self.storage, llm_callback=llm_callback)

        # Default to stdin/stdout if no callbacks
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

    # ===================
    # Sub-commands
    # ===================

    def run_new(
        self,
        domain: str,
        name: str = None,
        template: str = None,
        interactive: bool = True
    ) -> Optional[InterviewContext]:
        """
        Start a new interview.

        Args:
            domain: The domain to explore
            name: Optional initiative name
            template: Optional template scaffold (bootstrap, enterprise, etc.)
            interactive: If True, run interactive interview; if False, just create

        Returns:
            InterviewContext if completed, None if aborted
        """
        # Check for similar initiatives
        found_similar, similar = self.engine.check_continuation(domain)

        if found_similar and interactive:
            init = similar['initiative']
            similarity = similar['similarity']

            self.output_callback(f"""
I found a similar initiative from your previous work:

  Name: {init['name']}
  Domain: {init['original_domain']}
  Status: {init['status']}
  Similarity: {similarity:.0%}
  Last Updated: {init['last_updated']}

Would you like to continue that exploration or start fresh?
""")
            choice = self.input_callback("Enter 'continue' or 'new': ").lower()

            if choice == 'continue':
                return self.run_continue(init['id'])

        # Detect template if not specified
        if not template:
            detected = self.engine.detect_template(domain)
            if detected and interactive:
                self.output_callback(f"\nBased on your domain, the {detected.value.upper()} template might apply.")
                use_template = self.input_callback("Use this template as a starting scaffold? (y/n): ").lower()
                if use_template == 'y':
                    template = detected.value

        # Parse template
        template_enum = None
        if template:
            try:
                template_enum = TemplateScaffold(template.lower())
            except ValueError:
                self.output_callback(f"Unknown template: {template}. Proceeding without scaffold.")

        # Start interview
        state = self.engine.start_interview(
            domain=domain,
            initiative_name=name,
            template=template_enum
        )

        if interactive:
            return self._run_interactive_interview(state)
        else:
            self.output_callback(f"Initiative created: {state.initiative_id}")
            return state.context

    def run_continue(self, initiative_id: str = None) -> Optional[InterviewContext]:
        """
        Continue an existing interview or show list to choose from.

        Args:
            initiative_id: Specific initiative to continue (or None to list)

        Returns:
            InterviewContext if completed, None if aborted
        """
        if initiative_id:
            state = self.engine.resume_interview(initiative_id)
            if not state:
                self.output_callback(f"Initiative not found: {initiative_id}")
                return None
            return self._run_interactive_interview(state)

        # List draft initiatives
        drafts = self.storage.list_initiatives(status=InitiativeStatus.DRAFT)

        if not drafts:
            self.output_callback("No in-progress interviews found.")
            return None

        self.output_callback("\nIn-progress interviews:\n")
        for i, init in enumerate(drafts, 1):
            self.output_callback(f"  {i}. [{init['id'][:8]}] {init['name']}")
            self.output_callback(f"     Domain: {init['original_domain'][:60]}...")
            self.output_callback(f"     Last Updated: {init['last_updated']}\n")

        choice = self.input_callback("Enter number to continue (or 'q' to quit): ")

        if choice.lower() == 'q':
            return None

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(drafts):
                return self.run_continue(drafts[idx]['id'])
            else:
                self.output_callback("Invalid selection.")
                return None
        except ValueError:
            # Maybe they entered an ID directly
            if len(choice) >= 8:
                matches = [d for d in drafts if d['id'].startswith(choice)]
                if matches:
                    return self.run_continue(matches[0]['id'])
            self.output_callback("Invalid selection.")
            return None

    def run_list(
        self,
        status: str = None,
        limit: int = 20,
        format: str = "table"
    ) -> List[dict]:
        """
        List all initiatives.

        Args:
            status: Filter by status (draft, ready, active, archived)
            limit: Maximum number to show
            format: Output format (table, json)

        Returns:
            List of initiative dictionaries
        """
        status_enum = None
        if status:
            try:
                status_enum = InitiativeStatus(status.lower())
            except ValueError:
                self.output_callback(f"Unknown status: {status}")
                return []

        initiatives = self.storage.list_initiatives(status=status_enum, limit=limit)

        if format == "json":
            self.output_callback(json.dumps(initiatives, indent=2, default=str))
        else:
            if not initiatives:
                self.output_callback("No initiatives found.")
            else:
                self.output_callback(f"\n{'ID':<10} {'Status':<10} {'Name':<40} {'Updated':<20}")
                self.output_callback("-" * 80)
                for init in initiatives:
                    self.output_callback(
                        f"{init['id'][:8]:<10} "
                        f"{init['status']:<10} "
                        f"{init['name'][:38]:<40} "
                        f"{init['last_updated'][:19]:<20}"
                    )
                self.output_callback(f"\nTotal: {len(initiatives)} initiatives")

        return initiatives

    def run_export(
        self,
        initiative_id: str,
        output_path: str = None
    ) -> Optional[str]:
        """
        Export initiative context as Markdown.

        Args:
            initiative_id: Initiative to export (can be partial ID)
            output_path: Optional output file path

        Returns:
            Path to exported file, or None if failed
        """
        # Find initiative by partial ID
        initiatives = self.storage.list_initiatives()
        matches = [i for i in initiatives if i['id'].startswith(initiative_id)]

        if not matches:
            self.output_callback(f"No initiative found matching: {initiative_id}")
            return None

        if len(matches) > 1:
            self.output_callback(f"Multiple matches found. Please be more specific:")
            for m in matches:
                self.output_callback(f"  {m['id']}: {m['name']}")
            return None

        full_id = matches[0]['id']

        if output_path:
            path = self.storage.export_to_file(full_id, output_path)
        else:
            # Print to stdout
            markdown = self.storage.export_markdown(full_id)
            if markdown:
                self.output_callback(markdown)
                return "stdout"
            else:
                self.output_callback(f"Failed to export initiative: {full_id}")
                return None

        if path:
            self.output_callback(f"Exported to: {path}")
            return path
        else:
            self.output_callback(f"Failed to export initiative: {full_id}")
            return None

    def run_delete(
        self,
        initiative_id: str,
        force: bool = False
    ) -> bool:
        """
        Delete an initiative.

        Args:
            initiative_id: Initiative to delete (can be partial ID)
            force: Skip confirmation if True

        Returns:
            True if deleted, False otherwise
        """
        # Find initiative by partial ID
        initiatives = self.storage.list_initiatives()
        matches = [i for i in initiatives if i['id'].startswith(initiative_id)]

        if not matches:
            self.output_callback(f"No initiative found matching: {initiative_id}")
            return False

        if len(matches) > 1:
            self.output_callback(f"Multiple matches found. Please be more specific:")
            for m in matches:
                self.output_callback(f"  {m['id']}: {m['name']}")
            return False

        full_id = matches[0]['id']
        name = matches[0]['name']

        if not force:
            self.output_callback(f"\nAbout to delete: {name}")
            self.output_callback(f"ID: {full_id}")
            self.output_callback("This will remove all interview data for this initiative.")
            confirm = self.input_callback("Are you sure? (yes/no): ").lower()

            if confirm != 'yes':
                self.output_callback("Cancelled.")
                return False

        success = self.storage.delete_initiative(full_id)

        if success:
            self.output_callback(f"Deleted: {name}")
        else:
            self.output_callback(f"Failed to delete initiative: {full_id}")

        return success

    # ===================
    # Interactive Interview
    # ===================

    def _run_interactive_interview(self, state: InterviewState) -> Optional[InterviewContext]:
        """
        Run an interactive interview session.

        Args:
            state: The interview state to work with

        Returns:
            InterviewContext if completed, None if interrupted
        """
        # Show welcome
        welcome = self.engine.get_welcome_message(state)
        self.output_callback(f"\n{welcome}\n")

        current_dimension = None

        try:
            while True:
                # Get next question
                question = self.engine.get_next_question(state)

                if question is None:
                    # Interview complete
                    break

                # Show dimension transition if changed
                if current_dimension and current_dimension != question.dimension:
                    transition = self.engine.get_dimension_transition_message(
                        current_dimension, question.dimension
                    )
                    self.output_callback(f"\n{transition}\n")

                current_dimension = question.dimension

                # Ask question
                self.output_callback(f"\n{question.text}")
                response = self.input_callback("\nYour answer: ")

                # Handle special commands
                if response.lower() in ['quit', 'exit', 'q']:
                    self.output_callback("\nSaving progress and exiting...")
                    self.engine.save_interruption(state, "user_quit")
                    return None

                if response.lower() in ['skip', 'next']:
                    self.output_callback("Skipping this question.")
                    continue

                if response.lower() in ['done', 'ready', 'finish']:
                    self.output_callback("\nWrapping up the interview...")
                    break

                # Process response
                should_continue, progress_msg = self.engine.process_response(
                    state, question, response
                )

                # Show progress
                if progress_msg and state.progress.questions_asked % 3 == 0:
                    self.output_callback(f"\n[{progress_msg}]")

                # Check for follow-up
                if should_continue and len(response) > 50:
                    follow_up = self.engine.get_follow_up_question(
                        state, response, question.dimension
                    )
                    if follow_up:
                        self.output_callback(f"\n{follow_up.text}")
                        follow_response = self.input_callback("\nYour answer: ")

                        if follow_response and follow_response.lower() not in ['skip', 'next']:
                            self.engine.process_response(
                                state, follow_up, follow_response
                            )

                if not should_continue:
                    break

            # Complete interview
            context = self.engine.complete_interview(state)

            # Show completion
            completion_msg = self.engine.get_completion_message(state)
            self.output_callback(f"\n{completion_msg}\n")

            # Allow corrections
            corrections = self.input_callback("\nAny corrections? (or press Enter to confirm): ")
            if corrections:
                self.output_callback("Noted. You can update the context by running the interview again with --continue.")

            return context

        except KeyboardInterrupt:
            self.output_callback("\n\nInterrupted. Saving progress...")
            self.engine.save_interruption(state, "keyboard_interrupt")
            return None

    # ===================
    # Stats Command
    # ===================

    def run_stats(self) -> dict:
        """Show storage statistics."""
        stats = self.storage.get_stats()
        self.output_callback("\nInterview Storage Statistics:")
        self.output_callback(f"  Total Initiatives: {stats['total_initiatives']}")
        self.output_callback(f"    - Draft: {stats['draft_initiatives']}")
        self.output_callback(f"    - Ready: {stats['ready_initiatives']}")
        self.output_callback(f"  Total Sessions: {stats['total_sessions']}")
        self.output_callback(f"  Total Responses: {stats['total_responses']}")
        self.output_callback(f"  Open Gaps: {stats['open_gaps']}")
        self.output_callback(f"  Qdrant Vectors: {stats['qdrant_vectors']}")
        return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Universal Interview - Adaptive domain interview for ideation context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "sustainable packaging"          Start new interview
  %(prog)s --continue                        Resume in-progress interview
  %(prog)s --list                            List all initiatives
  %(prog)s --list --status ready             List ready initiatives
  %(prog)s --export abc123                   Export initiative as Markdown
  %(prog)s --delete abc123                   Delete an initiative
  %(prog)s --stats                           Show statistics
        """
    )

    # Positional argument for domain
    parser.add_argument(
        'domain',
        nargs='?',
        help="Domain to explore (starts new interview)"
    )

    # Sub-command flags
    parser.add_argument(
        '--continue', '-c',
        action='store_true',
        dest='do_continue',
        help="Continue an in-progress interview"
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help="List all initiatives"
    )

    parser.add_argument(
        '--export', '-e',
        metavar='ID',
        help="Export initiative context as Markdown"
    )

    parser.add_argument(
        '--delete', '-d',
        metavar='ID',
        help="Delete an initiative"
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help="Show storage statistics"
    )

    # Options
    parser.add_argument(
        '--name', '-n',
        help="Name for new initiative"
    )

    parser.add_argument(
        '--template', '-t',
        choices=['bootstrap', 'enterprise', 'regulated', 'sustainable'],
        help="Template scaffold to apply"
    )

    parser.add_argument(
        '--status', '-s',
        choices=['draft', 'ready', 'active', 'archived'],
        help="Filter list by status"
    )

    parser.add_argument(
        '--output', '-o',
        help="Output file path for export"
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help="Skip confirmation for delete"
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help="Output in JSON format"
    )

    parser.add_argument(
        '--initiative-id', '-i',
        help="Specific initiative ID for continue"
    )

    args = parser.parse_args()

    # Initialize runner
    runner = InterviewRunner()

    # Dispatch to appropriate command
    if args.stats:
        runner.run_stats()

    elif args.list:
        format_type = "json" if args.json else "table"
        runner.run_list(status=args.status, format=format_type)

    elif args.export:
        runner.run_export(args.export, args.output)

    elif args.delete:
        runner.run_delete(args.delete, args.force)

    elif args.do_continue:
        runner.run_continue(args.initiative_id)

    elif args.domain:
        result = runner.run_new(
            domain=args.domain,
            name=args.name,
            template=args.template
        )
        if result:
            print(f"\nInterview complete! Initiative ID: {result.initiative_id}")
            print("Use this context with: /universal-ideation-v3")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
