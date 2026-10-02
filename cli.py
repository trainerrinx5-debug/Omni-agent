"""Command-line interface for Omni Agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .agent import Agent
from .config import Settings
from .llm import OpenAICompatibleClient, ProviderError
from .memory import MemoryError, MemoryStore
from .safety import WriteApproval
from .tools import build_tool_registry


HELP_TEXT = """Commands:
  /help                 Show this help
  /tools                List enabled tools
  /memory               List user-managed memory notes
  /remember TEXT        Save a local note (confirmation required)
  /forget ID            Delete a memory note
  /clear                Clear this session's chat history
  /exit                 Quit

Memory is never saved automatically. Notes are sent to your configured model
provider as context on later turns, so do not store secrets or sensitive data.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omni-agent",
        description="A bounded, tool-using assistant for an explicitly scoped local workspace.",
    )
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL (or set OMNI_BASE_URL)")
    parser.add_argument("--model", help="Model name (or set OMNI_MODEL)")
    parser.add_argument("--workspace", help="Agent file-tool root (default: ./workspace)")
    parser.add_argument("--state-dir", help="Local state directory (default: ./.omni)")
    parser.add_argument("--max-steps", type=int, help="Maximum tool rounds per request (1-12)")
    parser.add_argument("--yes", action="store_true", help="Auto-approve workspace writes; use only for a trusted model in a disposable workspace")
    parser.add_argument("--prompt", help="Run one request and exit")
    parser.add_argument("--doctor", action="store_true", help="Print effective configuration without contacting providers")
    parser.add_argument("--list-tools", action="store_true", help="List enabled tool names without contacting a model")
    parser.add_argument("--version", action="version", version=f"omni-agent {__version__}")
    return parser


def _effective_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def _print_doctor(settings: Settings, *, auto_approve: bool) -> None:
    print("Omni Agent configuration (no network request made)")
    print(f"  Model endpoint: {_safe_url(settings.base_url)}")
    print(f"  Model:          {settings.model}")
    print(f"  Workspace:      {_effective_path(settings.workspace)}")
    print(f"  Memory file:    {_effective_path(settings.state_dir / 'memory.json')}")
    print(f"  Web search:     {'enabled (Brave key configured)' if settings.brave_search_api_key else 'disabled'}")
    print(f"  File writes:    {'auto-approved (--yes)' if auto_approve else 'approval required'}")


def _safe_url(value: str) -> str:
    # Configuration output intentionally omits any accidental credentials/query strings.
    from urllib.parse import urlsplit, urlunsplit

    try:
        parts = urlsplit(value)
        host = parts.hostname or "invalid"
        if parts.port:
            host += f":{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except ValueError:
        return "<invalid URL>"


def _print_tools(registry) -> None:
    for item in registry.openai_definitions():
        function = item["function"]
        print(f"  {function['name']}: {function['description']}")


def _handle_command(line: str, *, memory: MemoryStore, history: list[dict], registry) -> tuple[bool, list[dict]]:
    command, _, argument = line.partition(" ")
    command = command.casefold()
    argument = argument.strip()
    if command in {"/exit", "/quit"}:
        return False, history
    if command == "/help":
        print(HELP_TEXT)
    elif command == "/tools":
        _print_tools(registry)
    elif command == "/clear":
        history = []
        print("Session history cleared.")
    elif command == "/memory":
        notes = memory.list_notes()
        if not notes:
            print("No saved notes.")
        else:
            for note in notes:
                print(f"  {note['id']}  {note['text']}")
    elif command == "/remember":
        if not argument:
            print("Usage: /remember TEXT")
        elif not sys.stdin.isatty():
            print("Memory changes require an interactive terminal.")
        else:
            print("This note stays on disk locally, but is sent to the configured model provider on later turns. Do not save secrets or sensitive data.")
            try:
                answer = input("Save this note? [y/N] ").strip().casefold()
                if answer in {"y", "yes"}:
                    note = memory.add(argument)
                    print(f"Saved memory note {note['id']}.")
                else:
                    print("Not saved.")
            except MemoryError as exc:
                print(f"Memory error: {exc}")
    elif command == "/forget":
        if not argument:
            print("Usage: /forget ID")
        else:
            try:
                deleted = memory.delete(argument)
                print("Memory note deleted." if deleted else "No note found with that ID.")
            except MemoryError as exc:
                print(f"Memory error: {exc}")
    else:
        print("Unknown command. Use /help.")
    return True, history


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env(
            base_url=args.base_url,
            model=args.model,
            workspace=args.workspace,
            state_dir=args.state_dir,
            max_steps=args.max_steps,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if not settings.base_url or not settings.model:
        parser.error("model endpoint and model name must not be empty")
    try:
        client = OpenAICompatibleClient(
            settings.base_url,
            settings.model,
            api_key=settings.api_key,
            timeout_seconds=settings.timeout_seconds,
        )
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    if args.doctor:
        _print_doctor(settings, auto_approve=args.yes)
        return 0

    approval = WriteApproval(auto_approve=args.yes)
    try:
        registry = build_tool_registry(
            settings.workspace,
            approval,
            brave_search_api_key=settings.brave_search_api_key,
        )
        memory = MemoryStore(settings.state_dir / "memory.json")
    except (OSError, MemoryError, ValueError) as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 2

    if args.list_tools:
        _print_tools(registry)
        return 0

    agent = Agent(client, registry, memory=memory, max_steps=settings.max_steps)

    if args.prompt is not None:
        try:
            answer, _ = agent.run_turn(args.prompt)
            print(answer)
            return 0
        except (ProviderError, RuntimeError, ValueError) as exc:
            print(f"Agent error: {exc}", file=sys.stderr)
            return 1

    print(f"Omni Agent {__version__} — model: {settings.model}")
    print(f"Workspace: {_effective_path(settings.workspace)}")
    print("Use /help for commands; /exit to quit. File writes require approval unless --yes is set.\n")
    history: list[dict] = []
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if not line.strip():
            continue
        if line.lstrip().startswith("/"):
            keep_going, history = _handle_command(line.strip(), memory=memory, history=history, registry=registry)
            if not keep_going:
                print("Goodbye.")
                return 0
            continue
        try:
            answer, history = agent.run_turn(line, history)
            print(f"\nagent> {answer}\n")
        except ProviderError as exc:
            print(f"\nProvider error: {exc}\nCheck the endpoint/model and confirm the provider is running.\n")
        except (RuntimeError, ValueError) as exc:
            print(f"\nAgent error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
