"""
Incremental sync script for Mnemosyne.

Runs as a standalone process (no LLM). Called by:
  - mnemosyne setup (initial full sync)
  - hermes-agent cron job (incremental, every 30 min)

Supports two sources:
  - openclaw:  ~/.openclaw/agents/*/sessions/*.jsonl
  - hermes:    ~/.hermes/state.db sessions

Usage:
  python -m mnemosyne.scripts.sync --source openclaw
  python -m mnemosyne.scripts.sync --source hermes
  python -m mnemosyne.scripts.sync --source all        # first-run full sync
  python -m mnemosyne.scripts.sync --source all --check  # dry run: just count what's new
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add package root to path for imports
PACKAGE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from mnemosyne.core.memory import MnemosyneMemory
from mnemosyne.core.config import MnemosyneConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sync")


# ---------------------------------------------------------------------------
# Source: OpenClaw
# ---------------------------------------------------------------------------

def _get_last_sync_timestamp(data_dir: Path) -> float | None:
    """Load last_sync timestamp from mnemosyne.db metadata table."""
    import sqlite3
    db_path = data_dir / "mnemosyne.db"
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'last_openclaw_sync'"
        ).fetchone()
        conn.close()
        return float(row[0]) if row else None
    except Exception:
        return None


def _set_last_sync_timestamp(data_dir: Path, ts: float) -> None:
    import sqlite3
    db_path = data_dir / "mnemosyne.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_openclaw_sync', ?)",
        (str(ts),),
    )
    conn.commit()
    conn.close()


def _build_openclaw_session_id(file_path: Path, agent_name: str, session_id: str) -> str:
    return f"openclaw_{agent_name}_{session_id}"


def sync_openclaw(
    agents_dir: Path | None = None,
    since: float | None = None,
    limit: int | None = None,
    verbose: bool = False,
) -> dict:
    """
    Sync OpenClaw sessions modified since `since` timestamp.
    If `since` is None, performs a full scan (no dedup — relies on skip_existing).
    """
    import json
    from pathlib import Path

    if agents_dir is None:
        agents_dir = Path.home() / ".openclaw" / "agents"
    if not agents_dir.exists():
        logger.warning(f"OpenClaw agents dir not found: {agents_dir}")
        return {"total": 0, "indexed": 0, "skipped": 0, "failed": 0, "filtered": 0}

    # Agents to skip (setup noise)
    SKIP_AGENTS = {
        "cook", "council", "cipher", "gear", "monitor", "sentinel",
        "skeptic", "bench-minimax-minimax-m2-1", "ux-voice"
    }

    # Setup noise patterns
    SETUP_PATTERNS = [
        "check emails, calendar, weather",
        "heartbeat_ok",
        "check my email",
        "check calendar",
        "system check",
        "setup openclaw",
        "configure",
        "install dependencies",
        "pip install",
        "npm install",
    ]

    stats = {"total": 0, "indexed": 0, "skipped": 0, "failed": 0, "filtered": 0}
    memory = MnemosyneMemory()
    if not memory.initialize():
        logger.error("Failed to initialize Mnemosyne")
        return stats

    try:
        # Pre-fetch existing IDs once
        existing_ids: set[str] = set()
        try:
            result = memory.vector_store.conn.execute(
                "SELECT session_id FROM session_meta"
            ).fetchall()
            existing_ids = {row[0] for row in result}
            logger.info(f"Already indexed: {len(existing_ids)} sessions")
        except Exception as e:
            logger.warning(f"Could not check existing sessions: {e}")

        t0 = time.time()
        for agent_dir in agents_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            agent_name = agent_dir.name
            if agent_name in SKIP_AGENTS:
                continue

            sessions_dir = agent_dir / "sessions"
            if not sessions_dir.exists():
                continue

            for session_file in sessions_dir.glob("*.jsonl"):
                if ".deleted." in session_file.name or ".reset." in session_file.name:
                    continue

                # Incremental: skip files with old mtime
                if since is not None and session_file.stat().st_mtime <= since:
                    stats["skipped"] += 1
                    continue

                stats["total"] += 1
                if limit and stats["indexed"] >= limit:
                    break

                try:
                    messages = []
                    session_meta = {}
                    file_changed = False

                    with open(session_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            event = json.loads(line)
                            event_type = event.get("type")

                            if event_type == "session":
                                session_meta = {
                                    "id": event.get("id"),
                                    "timestamp": event.get("timestamp"),
                                    "cwd": event.get("cwd"),
                                }
                                file_changed = True
                            elif event_type == "message":
                                msg_data = event.get("message", {})
                                role = msg_data.get("role", "")
                                content = msg_data.get("content", "")
                                if isinstance(content, list):
                                    text_parts = [
                                        p.get("text", "")
                                        for p in content
                                        if isinstance(p, dict) and p.get("type") == "text"
                                    ]
                                    content = " ".join(text_parts)
                                if content and role in ("user", "assistant"):
                                    messages.append({
                                        "role": role,
                                        "content": content,
                                        "timestamp": event.get("timestamp"),
                                        "id": event.get("id"),
                                    })
                                    file_changed = True

                    if not messages or not session_meta.get("id"):
                        stats["skipped"] += 1
                        continue

                    session_id_str = _build_openclaw_session_id(
                        session_file, agent_name, session_meta["id"]
                    )

                    if session_id_str in existing_ids:
                        stats["skipped"] += 1
                        continue

                    # Filter setup noise
                    if messages:
                        first_user = next(
                            (m["content"].lower() for m in messages if m.get("role") == "user"),
                            "",
                        )
                        if first_user:
                            is_setup = any(p in first_user for p in SETUP_PATTERNS)
                            if is_setup:
                                user_msgs = [m for m in messages if m.get("role") == "user"]
                                if len(user_msgs) < 2 or len(user_msgs[0].get("content", "")) < 100:
                                    stats["filtered"] += 1
                                    continue

                    metadata = {
                        "source": "openclaw",
                        "agent": agent_name,
                        "started_at": session_meta.get("timestamp"),
                        "message_count": len(messages),
                        "cwd": session_meta.get("cwd", ""),
                        "original_id": session_meta["id"],
                    }

                    if memory.index_session(session_id_str, messages, metadata):
                        stats["indexed"] += 1
                        existing_ids.add(session_id_str)
                    else:
                        stats["failed"] += 1

                    if verbose and stats["indexed"] % 50 == 0 and stats["indexed"] > 0:
                        elapsed = time.time() - t0
                        rate = stats["indexed"] / elapsed
                        logger.info(
                            f"  Indexed {stats['indexed']} sessions "
                            f"({rate:.1f}/sec, {stats['skipped']} skipped)"
                        )

                except Exception as e:
                    logger.error(f"Failed to process {session_file}: {e}")
                    stats["failed"] += 1

                if limit and stats["indexed"] >= limit:
                    break

    finally:
        memory.close()

    elapsed = time.time() - t0
    logger.info(
        f"OpenClaw sync done: {stats['indexed']} indexed, "
        f"{stats['skipped']} skipped, {stats['filtered']} filtered, "
        f"{stats['failed']} failed in {elapsed:.1f}s"
    )
    return stats


# ---------------------------------------------------------------------------
# Source: Hermes SQLite
# ---------------------------------------------------------------------------

HERMES_SQLITE_QUERY = """
SELECT
    s.id,
    s.platform,
    s.user_id,
    s.thread_id,
    s.created_at,
    s.updated_at,
    GROUP_CONCAT(m.content, '\n---\n') AS content
FROM sessions s
LEFT JOIN messages m ON m.session_id = s.id
WHERE s.platform IN ('telegram', 'cli', 'skippy')
GROUP BY s.id
HAVING content IS NOT NULL AND content != ''
ORDER BY s.updated_at DESC
LIMIT ?
OFFSET ?
"""


def sync_hermes(
    db_path: Path | None = None,
    since: datetime | None = None,
    batch: int = 100,
    verbose: bool = False,
) -> dict:
    """Sync sessions from hermes-agent's SQLite state.db."""
    import sqlite3
    from datetime import datetime as dt

    if db_path is None:
        db_path = Path.home() / ".hermes" / "state.db"
    if not db_path.exists():
        logger.warning(f"Hermes state.db not found: {db_path}")
        return {"total": 0, "indexed": 0, "skipped": 0, "failed": 0}

    stats = {"total": 0, "indexed": 0, "skipped": 0, "failed": 0}
    memory = MnemosyneMemory()
    if not memory.initialize():
        logger.error("Failed to initialize Mnemosyne")
        return stats

    try:
        # Pre-fetch existing IDs
        existing_ids: set[str] = set()
        try:
            result = memory.vector_store.conn.execute(
                "SELECT session_id FROM session_meta"
            ).fetchall()
            existing_ids = {row[0] for row in result}
        except Exception:
            pass

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        offset = 0

        while True:
            rows = conn.execute(
                HERMES_SQLITE_QUERY,
                (batch, offset),
            ).fetchall()
            if not rows:
                break

            for row in rows:
                stats["total"] += 1
                session_id = f"hermes_{row['platform']}_{row['id']}"

                if session_id in existing_ids:
                    stats["skipped"] += 1
                    continue

                content = row["content"]
                if not content or len(content.strip()) < 50:
                    stats["skipped"] += 1
                    continue

                # Parse messages (simple heuristic: split on ---\n markers)
                messages = []
                for i, chunk in enumerate(content.split("\n---\n")):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    role = "user" if i % 2 == 0 else "assistant"
                    messages.append({"role": role, "content": chunk})

                metadata = {
                    "source": "hermes",
                    "platform": row["platform"],
                    "user_id": row["user_id"],
                    "thread_id": row["thread_id"],
                    "started_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }

                if memory.index_session(session_id, messages, metadata):
                    stats["indexed"] += 1
                    existing_ids.add(session_id)
                else:
                    stats["failed"] += 1

            offset += batch
            conn.commit()

            if verbose:
                logger.info(f"  Hermes: processed {offset} rows, {stats['indexed']} indexed")

    finally:
        conn.close()
        memory.close()

    logger.info(
        f"Hermes sync done: {stats['indexed']} indexed, "
        f"{stats['skipped']} skipped, {stats['failed']} failed"
    )
    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mnemosyne incremental sync — index new sessions since last run"
    )
    parser.add_argument(
        "--source",
        choices=["openclaw", "hermes", "all"],
        default="all",
        help="Which source to sync (default: all)",
    )
    parser.add_argument(
        "--agents-dir",
        type=Path,
        default=None,
        help="Path to .openclaw/agents (default: ~/.openclaw/agents)",
    )
    parser.add_argument(
        "--hermes-db",
        type=Path,
        default=None,
        help="Path to hermes state.db (default: ~/.hermes/state.db)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Dry run: count new sessions without indexing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max sessions to index per source (default: unlimited)",
    )
    parser.add_argument(
        "--since",
        type=float,
        default=None,
        help="Only process files modified after this Unix timestamp",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = MnemosyneConfig()
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check mode: just report what would be synced
    if args.check:
        last_ts = _get_last_sync_timestamp(data_dir)
        if last_ts:
            from datetime import datetime as dt
            dt_str = dt.fromtimestamp(last_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
            logger.info(f"Last sync: {dt_str} (Unix {last_ts})")
        else:
            logger.info("No previous sync found — full sync needed")
        return 0

    # Incremental vs full
    if args.source in ("openclaw", "all"):
        last_ts = _get_last_sync_timestamp(data_dir)
        sync_ts = args.since if args.since is not None else last_ts
        logger.info(
            f"Syncing OpenClaw (since={sync_ts or 'full scan'})"
        )
        result = sync_openclaw(
            agents_dir=args.agents_dir,
            since=sync_ts,
            limit=args.limit,
            verbose=args.verbose,
        )
        # Record successful sync timestamp
        if result["indexed"] > 0 or result["total"] > 0:
            _set_last_sync_timestamp(data_dir, time.time())

    if args.source in ("hermes", "all"):
        logger.info("Syncing Hermes sessions")
        result = sync_hermes(
            db_path=args.hermes_db,
            verbose=args.verbose,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
