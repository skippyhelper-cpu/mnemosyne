#!/usr/bin/env python3
"""
Mnemosyne MCP Server — exposes MnemosyneMemory as an MCP tool server.

Provides tools:
  - search: hybrid search with three-tier self-learning
  - synthesize: trigger Tier 3 pattern synthesis
  - cache_stats: return cache and hot-vector statistics
  - invalidate_cache: invalidate cache for a session

Run directly:
    python -m mnemosyne.core.mcp_server

Or configure in config.yaml as an MCP server.
"""

import sys
import os
import logging
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

APP_NAME = "mnemosyne"
APP_VERSION = "1.1.0"

_memory = None


def get_memory():
    """Lazily initialize and return MnemosyneMemory singleton."""
    global _memory
    if _memory is None:
        from mnemosyne.core.memory import MnemosyneMemory

        _memory = MnemosyneMemory()
        if not _memory.initialize():
            raise RuntimeError("Failed to initialize MnemosyneMemory")
    return _memory


server = Server(APP_NAME)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search",
            description=(
                "Hybrid search with three-tier self-learning. "
                "Tier 1: query cache with SHAKE-256 receipts. "
                "Tier 2: hot-vector score boosting based on query frequency. "
                "Tier 3: canonical entries from pattern synthesis. "
                "Returns results, cache source, cache stats, and hot-session boost info."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "session_key": {
                        "type": "string",
                        "description": "Session identifier for cache scoping (default: 'default')",
                        "default": "default"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 100
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="synthesize",
            description=(
                "Trigger Tier 3 pattern synthesis: find recurring topics across sessions "
                "and create canonical knowledge entries."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "min_topic_frequency": {
                        "type": "integer",
                        "description": "Minimum topic frequency in Kuzu graph (default: 3)",
                        "default": 3,
                        "minimum": 1
                    },
                    "min_session_count": {
                        "type": "integer",
                        "description": "Minimum sessions discussing topic (default: 3)",
                        "default": 3,
                        "minimum": 2
                    },
                },
            },
        ),
        Tool(
            name="cache_stats",
            description="Return current query cache and hot-vector statistics.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="invalidate_cache",
            description="Invalidate all cache entries for a session.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_key": {
                        "type": "string",
                        "description": "Session key to invalidate (default: 'default')",
                        "default": "default"
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        memory = get_memory()

        if name == "search":
            result = memory.search_with_learning(
                query=str(arguments["query"]),
                session_key=str(arguments.get("session_key", "default")),
                top_k=int(arguments.get("top_k", 5)),
            )
            return [TextContent(type="text", text=_format_search(result))]

        elif name == "synthesize":
            created = memory.sl.synthesize_patterns(
                min_topic_freq=int(arguments.get("min_topic_frequency", 3)),
                min_sessions=int(arguments.get("min_session_count", 3)),
            )
            if not created:
                return [TextContent(
                    type="text",
                    text="No canonical entries created (no recurring topics met the threshold)."
                )]
            lines = [f"Synthesized {len(created)} canonical entries:"]
            for entry in created:
                preview = (entry.content or "")[:80]
                lines.append(
                    f"  • {entry.topic} "
                    f"(confidence={entry.confidence:.2f}, "
                    f"sessions={len(entry.source_sessions)}) — {preview}"
                )
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "cache_stats":
            stats = memory.sl.get_all_stats()
            import json
            return [TextContent(type="text", text=json.dumps(stats, indent=2, default=str))]

        elif name == "invalidate_cache":
            session_key = str(arguments.get("session_key", "default"))
            evicted = memory.sl.query_cache.invalidate_session(session_key)
            return [TextContent(
                type="text",
                text=f"Invalidated {evicted} cache entries for session '{session_key}'"
            )]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logging.error("MCP tool '%s' failed: %s", name, e, exc_info=True)
        return [TextContent(type="text", text=f"Error: {e}")]


def _format_search(result: dict) -> str:
    source = result.get("source", "?")
    cache = result.get("cache_stats", {})
    boost = result.get("boost_info", {})
    items = result.get("results", [])

    hot = boost.get("hot_sessions", [])
    hot_line = f"Hot sessions: {', '.join(str(h) for h in hot[:5])}" if hot else "(no hot sessions)"

    lines = [
        f"Source: {source}  |  Cache: hits={cache.get('hits', '?')} "
        f"misses={cache.get('misses', '?')}  entries={cache.get('entries', '?')}",
        f"  {hot_line}",
        f"\nTop {len(items)} results:",
    ]

    for i, r in enumerate(items, 1):
        sid = r.get("session_id", "?")
        score = r.get("score", "?")
        preview = (r.get("content") or r.get("text_preview", ""))[:100]
        lines.append(f"  {i}. [{sid}] score={score:.3f} — {preview}")

    return "\n".join(lines)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Mnemosyne MCP server v%s", APP_VERSION)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
