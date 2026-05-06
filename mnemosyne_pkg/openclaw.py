"""Importer for OpenClaw JSONL session files."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from ..core.memory import MnemosyneMemory
from ..core.config import MnemosyneConfig

logger = logging.getLogger(__name__)


class OpenClawSessionImporter:
    """Import OpenClaw JSONL sessions into Mnemosyne."""

    # Agents to skip (setup/config noise)
    SKIP_AGENTS: Set[str] = {
        "cook", "council", "cipher", "gear", "monitor", "sentinel",
        "skeptic", "bench-minimax-minimax-m2-1", "ux-voice"
    }

    def __init__(
        self,
        agents_dir: Path,
        mnemosyne: Optional[MnemosyneMemory] = None,
        config: Optional[MnemosyneConfig] = None,
        skip_agents: Optional[Set[str]] = None
    ):
        self.agents_dir = Path(agents_dir)
        self.mnemosyne = mnemosyne or MnemosyneMemory(config)
        self.config = config or MnemosyneConfig()
        self.skip_agents = skip_agents or self.SKIP_AGENTS

    def iter_session_files(self) -> Iterator[Path]:
        """Iterate over all session files across agents."""
        if not self.agents_dir.exists():
            logger.error(f"Agents directory not found: {self.agents_dir}")
            return

        for agent_dir in self.agents_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            agent_name = agent_dir.name
            if agent_name in self.skip_agents:
                continue

            sessions_dir = agent_dir / "sessions"
            if not sessions_dir.exists():
                continue

            for session_file in sessions_dir.glob("*.jsonl"):
                if ".deleted." in session_file.name or ".reset." in session_file.name:
                    continue
                yield session_file

    def parse_session(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Parse a single OpenClaw session file."""
        try:
            messages = []
            session_meta = {}

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type")

                    if event_type == "session":
                        session_meta = {
                            "id": event.get("id"),
                            "timestamp": event.get("timestamp"),
                            "cwd": event.get("cwd"),
                            "version": event.get("version")
                        }

                    elif event_type == "message":
                        msg_data = event.get("message", {})
                        role = msg_data.get("role", "")
                        content = msg_data.get("content", "")

                        if isinstance(content, list):
                            text_parts = []
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    text_parts.append(part.get("text", ""))
                            content = " ".join(text_parts)

                        if content and role in ("user", "assistant"):
                            messages.append({
                                "role": role,
                                "content": content,
                                "timestamp": event.get("timestamp"),
                                "id": event.get("id")
                            })

                    elif event_type == "model_change":
                        messages.append({
                            "role": "system",
                            "content": f"[Model: {event.get('provider')}/{event.get('modelId')}]",
                            "timestamp": event.get("timestamp"),
                            "type": "model_change"
                        })

                    elif event_type == "custom":
                        custom_type = event.get("customType", "")
                        if custom_type == "tool-execution":
                            data = event.get("data", {})
                            tool_name = data.get("toolName", "unknown")
                            messages.append({
                                "role": "tool",
                                "content": f"[Tool: {tool_name}]",
                                "timestamp": event.get("timestamp"),
                                "tool_name": tool_name
                            })

            if not messages:
                return None

            agent_name = file_path.parent.parent.name

            return {
                "id": session_meta.get("id") or file_path.stem,
                "agent": agent_name,
                "timestamp": session_meta.get("timestamp"),
                "cwd": session_meta.get("cwd", ""),
                "messages": messages,
                "file_path": str(file_path)
            }

        except Exception as e:
            logger.error(f"Failed to parse session {file_path}: {e}")
            return None

    def filter_setup_noise(self, session: Dict[str, Any]) -> bool:
        """Filter out setup/configuration sessions."""
        messages = session.get("messages", [])
        if not messages:
            return False

        first_user = None
        for msg in messages:
            if msg.get("role") == "user":
                first_user = msg.get("content", "").lower()
                break

        if not first_user:
            return False

        setup_patterns = [
            "check emails, calendar, weather",
            "heartbeat_ok",
            "check my email",
            "check calendar",
            "system check",
            "setup openclaw",
            "configure",
            "install dependencies",
            "pip install",
            "npm install"
        ]

        for pattern in setup_patterns:
            if pattern in first_user:
                return False

        user_msgs = [m for m in messages if m.get("role") == "user"]
        if len(user_msgs) < 2:
            content = user_msgs[0].get("content", "") if user_msgs else ""
            if len(content) < 100:
                return False

        return True

    def import_session(self, session: Dict[str, Any]) -> bool:
        try:
            session_id = f"openclaw_{session['agent']}_{session['id']}"

            metadata = {
                "source": "openclaw",
                "agent": session["agent"],
                "started_at": session.get("timestamp"),
                "message_count": len(session.get("messages", [])),
                "cwd": session.get("cwd", ""),
                "original_id": session["id"]
            }

            return self.mnemosyne.index_session(
                session_id=session_id,
                messages=session["messages"],
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Failed to import session {session.get('id')}: {e}")
            return False

    def import_all(
        self,
        limit: Optional[int] = None,
        skip_existing: bool = True,
        filter_setup: bool = True
    ) -> Dict[str, int]:
        if not self.mnemosyne.initialize():
            logger.error("Failed to initialize Mnemosyne")
            return {"total": 0, "imported": 0, "failed": 0, "skipped": 0, "filtered": 0}

        stats = {"total": 0, "imported": 0, "failed": 0, "skipped": 0, "filtered": 0}

        try:
            # Get existing IDs
            existing_ids: Set[str] = set()
            if skip_existing and self.mnemosyne.vector_store:
                try:
                    result = self.mnemosyne.vector_store.conn.execute(
                        "SELECT session_id FROM session_meta"
                    ).fetchall()
                    existing_ids = {row[0] for row in result if row[0].startswith("openclaw_")}
                    logger.info(f"Found {len(existing_ids)} existing OpenClaw sessions")
                except Exception as e:
                    logger.warning(f"Could not check existing sessions: {e}")

            for session_file in self.iter_session_files():
                stats["total"] += 1

                if limit and stats["imported"] >= limit:
                    break

                session = self.parse_session(session_file)
                if not session:
                    stats["failed"] += 1
                    continue

                session_id = f"openclaw_{session['agent']}_{session['id']}"

                if skip_existing and session_id in existing_ids:
                    stats["skipped"] += 1
                    continue

                if filter_setup and not self.filter_setup_noise(session):
                    stats["filtered"] += 1
                    continue

                if self.import_session(session):
                    stats["imported"] += 1
                else:
                    stats["failed"] += 1

                if stats["total"] % 50 == 0:
                    logger.info(
                        f"Progress: {stats['total']} processed, "
                        f"{stats['imported']} imported, {stats['filtered']} filtered"
                    )

        except Exception as e:
            logger.error(f"Import failed: {e}")
        finally:
            self.mnemosyne.close()

        logger.info(
            f"Import complete: {stats['imported']} imported, "
            f"{stats['failed']} failed, {stats['filtered']} filtered"
        )
        return stats


def run_import(
    agents_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    verbose: bool = True,
    filter_setup: bool = True
) -> Dict[str, int]:
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    if agents_dir is None:
        agents_dir = Path.home() / ".openclaw" / "agents"

    if not agents_dir.exists():
        logger.error(f"OpenClaw agents directory not found: {agents_dir}")
        return {"total": 0, "imported": 0, "failed": 0, "skipped": 0, "filtered": 0}

    importer = OpenClawSessionImporter(agents_dir)

    total_files = sum(1 for _ in importer.iter_session_files())
    logger.info(f"Found {total_files} session files in {agents_dir}")

    return importer.import_all(limit=limit, filter_setup=filter_setup)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Import OpenClaw sessions to Mnemosyne")
    parser.add_argument("--agents-dir", type=Path, help="Path to .openclaw/agents")
    parser.add_argument("--limit", type=int, help="Max sessions to import")
    parser.add_argument("--no-filter", action="store_true", help="Don't filter setup sessions")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    stats = run_import(
        agents_dir=args.agents_dir,
        limit=args.limit,
        verbose=args.verbose,
        filter_setup=not args.no_filter
    )
    print(f"\nImport complete: {stats}")
