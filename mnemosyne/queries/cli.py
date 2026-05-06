"""CLI for querying and managing Mnemosyne memory."""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..core.memory import MnemosyneMemory
from ..core.config import MnemosyneConfig

logger = logging.getLogger(__name__)

PREFERRED_SOURCES = ["telegram", "cli", "skippy", "openclaw"]
EXCLUDED_SOURCES = ["cron"]


# ---------------------------------------------------------------------------
# Setup command
# ---------------------------------------------------------------------------

def cmd_setup(args):
    """Run post-install setup: initial sync + cron registration."""
    from ..scripts.post_install import run
    run()
    return 0


# ---------------------------------------------------------------------------
# Sync command
# ---------------------------------------------------------------------------

def cmd_sync(args):
    """Run incremental sync (or full sync if --full is passed)."""
    sync_script = Path(__file__).parent.parent / "scripts" / "sync.py"
    python_path = sys.executable

    cmd = [python_path, str(sync_script)]
    if args.source:
        cmd.extend(["--source", args.source])
    if args.verbose:
        cmd.append("--verbose")
    if args.check:
        cmd.append("--check")
    if args.full:
        # Force full scan: pass --since 0
        cmd.extend(["--since", "0"])

    result = subprocess.run(cmd)
    return result.returncode


# ---------------------------------------------------------------------------
# Search command
# ---------------------------------------------------------------------------

def cmd_search(args):
    memory = MnemosyneMemory()
    if not memory.initialize():
        print("Failed to initialize Mnemosyne", file=sys.stderr)
        return 1

    try:
        results = memory.search(
            query=args.query,
            top_k=args.limit,
            preferred_sources=PREFERRED_SOURCES,
            excluded_sources=EXCLUDED_SOURCES,
        )

        if not results:
            print("No results found.")
            return 0

        print(f"\nFound {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. Session: {result['session_id']}")
            print(f"   Score: {result.get('score', 0):.3f}")
            metadata = result.get('metadata', {})
            if 'source' in metadata:
                print(f"   Source: {metadata['source']}")
            print(f"   Preview: {result.get('text_preview', 'N/A')[:200]}...")
            print()
        return 0
    finally:
        memory.close()


# ---------------------------------------------------------------------------
# Context command
# ---------------------------------------------------------------------------

def cmd_context(args):
    memory = MnemosyneMemory()
    if not memory.initialize():
        print("Failed to initialize Mnemosyne", file=sys.stderr)
        return 1

    try:
        context = memory.get_session_context(args.session_id)

        if not context or 'error' in context:
            print(f"Session not found: {args.session_id}")
            return 1

        print(f"\nSession Context: {args.session_id}\n")
        print("Entities mentioned:")
        for entity in context.get('entities', []):
            print(f"  - {entity['name']} ({entity['type']})")

        print("\nRelated sessions:")
        for rel in context.get('related_sessions', []):
            print(f"  - {rel['session_id']} (strength: {rel['strength']})")

        if context.get('corrects'):
            print("\nCorrections made:")
            for corr in context['corrects']:
                print(f"  - {corr['field']}: {corr['old_value']} → {corr['new_value']}")

        if context.get('corrected_by'):
            print("\nCorrected by:")
            for corr in context['corrected_by']:
                print(f"  - {corr['session_id']}: {corr['field']}")

        return 0
    finally:
        memory.close()


# ---------------------------------------------------------------------------
# Stats command
# ---------------------------------------------------------------------------

def cmd_stats(args):
    memory = MnemosyneMemory()
    if not memory.initialize():
        print("Failed to initialize Mnemosyne", file=sys.stderr)
        return 1

    try:
        print("\nMnemosyne Memory Statistics\n")

        if memory.vector_store:
            vstats = memory.vector_store.get_stats()
            print("Vector Store (sqlite-vec + FTS5):")
            print(f"  Indexed sessions: {vstats.get('vectors_count', 0)}")
            print(f"  FTS entries: {vstats.get('fts_entries', 0)}")
            print(f"  Status: {vstats.get('status', 'unknown')}")

        if memory.kuzu:
            kstats = memory.kuzu.get_stats()
            print("\nKnowledge Graph (Kuzu):")
            print(f"  Sessions: {kstats.get('session', 0)}")
            print(f"  Entities: {kstats.get('entity', 0)}")
            print(f"  Topics: {kstats.get('topic', 0)}")

        if memory.sl:
            sl_stats = memory.sl.get_all_stats()
            print("\nSelf-Learning:")
            qs = sl_stats.get('query_cache', {})
            print(f"  Query cache: {qs.get('entries', 0)} entries, "
                  f"hit rate {qs.get('hit_rate', '0%')}")
            hv = sl_stats.get('hot_vectors', {})
            print(f"  Hot vectors: {hv.get('tracked_sessions', 0)} tracked sessions")

        print("\nData Paths:")
        print(f"  Data directory: {memory.config.data_dir}")
        print(f"  Graph store: {memory.config.kuzu_path}")

        return 0
    finally:
        memory.close()


# ---------------------------------------------------------------------------
# Import command (legacy)
# ---------------------------------------------------------------------------

def cmd_import(args):
    from ..importers.hermes_sqlite import run_import
    stats = run_import(
        sqlite_path=args.db,
        limit=args.limit,
        verbose=True
    )
    print(f"\nImport complete: {stats}")
    return 0 if stats['failed'] == 0 else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _is_setup_done() -> bool:
    """Check if setup has been run at least once."""
    try:
        config = MnemosyneConfig()
        marker = Path(config.data_dir) / ".setup_done"
        return marker.exists()
    except Exception:
        return False


def _mark_setup_done():
    """Mark setup as complete."""
    try:
        config = MnemosyneConfig()
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
        (Path(config.data_dir) / ".setup_done").touch()
    except Exception:
        pass


def _auto_setup():
    """
    Run setup automatically if it hasn't been done yet.
    Called on every CLI invocation — fast check, no redundant work.
    """
    if _is_setup_done():
        return

    logger.info("First run detected — running Mnemosyne setup...")
    from ..scripts import post_install
    try:
        post_install.run()
        _mark_setup_done()
        logger.info("Setup complete.\n")
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        logger.info("You can re-run setup manually: mnemosyne setup")


def main():
    # Auto-setup on first run (transparent to user)
    _auto_setup()

    parser = argparse.ArgumentParser(
        prog='mnemosyne',
        description='Mnemosyne - Local semantic memory for Hermes agents'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # setup
    setup_parser = subparsers.add_parser('setup', help='Run post-install setup (sync + cron)')
    setup_parser.set_defaults(func=cmd_setup)

    # sync
    sync_parser = subparsers.add_parser('sync', help='Sync new sessions (incremental)')
    sync_parser.add_argument('--source', choices=['openclaw', 'hermes', 'all'], default='all')
    sync_parser.add_argument('--full', action='store_true', help='Force full scan (no dedup)')
    sync_parser.add_argument('--check', action='store_true', help='Show sync status without importing')
    sync_parser.add_argument('-v', '--verbose', action='store_true')
    sync_parser.set_defaults(func=cmd_sync)

    # search
    search_parser = subparsers.add_parser('search', help='Search sessions')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('-l', '--limit', type=int, default=5)
    search_parser.set_defaults(func=cmd_search)

    # context
    ctx_parser = subparsers.add_parser('context', help='Get session context')
    ctx_parser.add_argument('session_id', help='Session ID')
    ctx_parser.set_defaults(func=cmd_context)

    # stats
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    stats_parser.set_defaults(func=cmd_stats)

    # import (legacy)
    import_parser = subparsers.add_parser('import', help='Import from Hermes SQLite')
    import_parser.add_argument('--db', type=Path)
    import_parser.add_argument('--limit', type=int)
    import_parser.set_defaults(func=cmd_import)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == '__main__':
    exit(main())
