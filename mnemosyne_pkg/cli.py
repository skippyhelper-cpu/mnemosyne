"""CLI for querying Mnemosyne memory."""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from ..core.memory import MnemosyneMemory
from ..core.config import MnemosyneConfig
from ..importers.hermes_sqlite import run_import

# Sources to include in default search (real conversations, not cron noise)
PREFERRED_SOURCES = ["telegram", "cli", "skippy", "openclaw"]
# Sources to always exclude from default search
EXCLUDED_SOURCES = ["cron"]

logger = logging.getLogger(__name__)


def cmd_search(args):
    memory = MnemosyneMemory()
    if not memory.initialize():
        print("Failed to initialize Mnemosyne", file=__import__('sys').stderr)
        return 1

    try:
        results = memory.search(query=args.query, top_k=args.limit, preferred_sources=PREFERRED_SOURCES, excluded_sources=EXCLUDED_SOURCES)

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


def cmd_context(args):
    memory = MnemosyneMemory()
    if not memory.initialize():
        print("Failed to initialize Mnemosyne", file=__import__('sys').stderr)
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


def cmd_stats(args):
    memory = MnemosyneMemory()
    if not memory.initialize():
        print("Failed to initialize Mnemosyne", file=__import__('sys').stderr)
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


def cmd_import(args):
    stats = run_import(
        sqlite_path=args.db,
        limit=args.limit,
        verbose=True
    )
    print(f"\nImport complete: {stats}")
    return 0 if stats['failed'] == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        prog='mnemosyne',
        description='Mnemosyne - Local semantic memory for Hermes'
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    search_parser = subparsers.add_parser('search', help='Search for sessions')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('-l', '--limit', type=int, default=5, help='Max results')
    search_parser.set_defaults(func=cmd_search)

    context_parser = subparsers.add_parser('context', help='Get session context')
    context_parser.add_argument('session_id', help='Session ID')
    context_parser.set_defaults(func=cmd_context)

    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    stats_parser.set_defaults(func=cmd_stats)

    import_parser = subparsers.add_parser('import', help='Import from Hermes SQLite')
    import_parser.add_argument('--db', type=Path, help='Path to state.db')
    import_parser.add_argument('--limit', type=int, help='Max sessions to import')
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
