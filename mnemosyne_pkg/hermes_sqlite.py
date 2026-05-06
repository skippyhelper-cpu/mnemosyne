"""Importer for Hermes SQLite sessions."""

import sqlite3
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from ..core.memory import MnemosyneMemory
from ..core.config import MnemosyneConfig

logger = logging.getLogger(__name__)


class HermesSQLiteImporter:
    """Import sessions from Hermes SQLite database into Mnemosyne."""

    def __init__(
        self,
        sqlite_path: Path,
        mnemosyne: Optional[MnemosyneMemory] = None,
        config: Optional[MnemosyneConfig] = None
    ):
        self.sqlite_path = sqlite_path
        self.mnemosyne = mnemosyne or MnemosyneMemory(config)
        self.config = config or MnemosyneConfig()

    def get_session_count(self) -> int:
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sessions")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.error(f"Failed to count sessions: {e}")
            return 0

    def iter_sessions(self, batch_size: int = 10) -> Iterator[Dict]:
        try:
            conn = sqlite3.connect(self.sqlite_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT s.id, s.source, s.model, s.started_at, s.message_count, s.system_prompt
                FROM sessions s
                ORDER BY s.started_at DESC
            """)

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                for row in rows:
                    yield {
                        "id": row["id"],
                        "source": row["source"],
                        "model": row["model"],
                        "started_at": row["started_at"],
                        "message_count": row["message_count"],
                        "system_prompt": row["system_prompt"]
                    }

            conn.close()

        except Exception as e:
            logger.error(f"Failed to iterate sessions: {e}")

    def get_session_messages(self, session_id: str) -> List[Dict]:
        try:
            conn = sqlite3.connect(self.sqlite_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT role, content, timestamp, tool_name, reasoning
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))

            messages = []
            for row in cursor.fetchall():
                msg = {"role": row["role"], "content": row["content"], "timestamp": row["timestamp"]}
                if row["tool_name"]:
                    msg["tool_name"] = row["tool_name"]
                if row["reasoning"]:
                    msg["reasoning"] = row["reasoning"]
                messages.append(msg)

            conn.close()
            return messages

        except Exception as e:
            logger.error(f"Failed to get messages for {session_id}: {e}")
            return []

    def import_session(self, session: Dict) -> bool:
        try:
            session_id = session["id"]
            messages = self.get_session_messages(session_id)
            if not messages:
                return False

            metadata = {
                "source": session.get("source", "unknown"),
                "model": session.get("model", ""),
                "started_at": session.get("started_at"),
                "message_count": session.get("message_count", len(messages)),
                "summary": self._generate_summary(messages)
            }

            return self.mnemosyne.index_session(
                session_id=session_id,
                messages=messages,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Failed to import session {session.get('id')}: {e}")
            return False

    def import_all(
        self,
        limit: Optional[int] = None,
        skip_existing: bool = True
    ) -> Dict[str, int]:
        if not self.mnemosyne.initialize():
            logger.error("Failed to initialize Mnemosyne")
            return {"total": 0, "imported": 0, "failed": 0, "skipped": 0}

        stats = {"total": 0, "imported": 0, "failed": 0, "skipped": 0}

        try:
            # Get existing IDs from vector store
            existing_ids = set()
            if skip_existing and self.mnemosyne.vector_store:
                try:
                    result = self.mnemosyne.vector_store.conn.execute(
                        "SELECT session_id FROM session_meta"
                    ).fetchall()
                    existing_ids = {row[0] for row in result}
                    logger.info(f"Found {len(existing_ids)} existing sessions")
                except Exception as e:
                    logger.warning(f"Could not check existing sessions: {e}")

            for session in self.iter_sessions():
                stats["total"] += 1

                if limit and stats["imported"] >= limit:
                    break

                if skip_existing and session["id"] in existing_ids:
                    stats["skipped"] += 1
                    continue

                if self.import_session(session):
                    stats["imported"] += 1
                else:
                    stats["failed"] += 1

                if stats["total"] % 50 == 0:
                    logger.info(
                        f"Progress: {stats['total']} total, "
                        f"{stats['imported']} imported, {stats['failed']} failed"
                    )

        except Exception as e:
            logger.error(f"Import failed: {e}")
        finally:
            self.mnemosyne.close()

        logger.info(f"Import complete: {stats['imported']} imported, {stats['failed']} failed")
        return stats

    def _generate_summary(self, messages: List[Dict], max_chars: int = 300) -> str:
        if not messages:
            return ""
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return ""
        first = user_msgs[0].get("content", "")[:150]
        if len(user_msgs) > 1:
            last = user_msgs[-1].get("content", "")[:150]
            return f"Started: {first}... Ended: {last}"
        return first[:max_chars]


def run_import(
    sqlite_path: Optional[Path] = None,
    limit: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, int]:
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    if sqlite_path is None:
        sqlite_path = Path.home() / ".hermes" / "state.db"

    if not sqlite_path.exists():
        logger.error(f"SQLite database not found: {sqlite_path}")
        return {"total": 0, "imported": 0, "failed": 0, "skipped": 0}

    importer = HermesSQLiteImporter(sqlite_path)
    total_count = importer.get_session_count()
    logger.info(f"Found {total_count} sessions in SQLite")

    return importer.import_all(limit=limit)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Import Hermes sessions to Mnemosyne")
    parser.add_argument("--db", type=Path, help="Path to state.db")
    parser.add_argument("--limit", type=int, help="Max sessions to import")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    stats = run_import(sqlite_path=args.db, limit=args.limit, verbose=args.verbose)
    print(f"\nImport complete: {stats}")
