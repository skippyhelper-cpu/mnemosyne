"""Kuzu graph database for knowledge relationships."""

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import MnemosyneConfig

logger = logging.getLogger(__name__)


class KuzuGraph:
    """Kuzu embedded graph database client."""

    def __init__(self, config: MnemosyneConfig):
        self.config = config
        self.db = None
        self.conn = None

    def initialize(self) -> bool:
        """Initialize Kuzu database with schema."""
        try:
            import kuzu

            logger.info(f"Initializing Kuzu at {self.config.kuzu_path}")

            # Ensure parent directory exists
            self.config.kuzu_path.parent.mkdir(parents=True, exist_ok=True)

            self.db = kuzu.Database(str(self.config.kuzu_path))
            self.conn = kuzu.Connection(self.db)

            self._create_schema()

            return True

        except ImportError:
            logger.error("Kuzu not installed. Run: pip install kuzu")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Kuzu: {e}")
            return False

    def _create_schema(self):
        """Create node and relationship tables."""
        try:
            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Session(
                    id STRING PRIMARY KEY,
                    source STRING,
                    started_at TIMESTAMP,
                    message_count INT64,
                    summary STRING
                )
            """)

            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Entity(
                    name STRING PRIMARY KEY,
                    type STRING,
                    context STRING
                )
            """)

            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Topic(
                    name STRING PRIMARY KEY,
                    frequency INT64 DEFAULT 1
                )
            """)

            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS MENTIONS(
                    FROM Session TO Entity,
                    MANY_MANY
                )
            """)

            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS DISCUSSES(
                    FROM Session TO Topic,
                    MANY_MANY
                )
            """)

            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS RELATED_TO(
                    FROM Session TO Session,
                    strength DOUBLE,
                    MANY_MANY
                )
            """)

            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS CORRECTS(
                    FROM Session TO Session,
                    field STRING,
                    old_value STRING,
                    new_value STRING,
                    MANY_MANY
                )
            """)

            logger.info("Kuzu schema created")

        except Exception as e:
            logger.debug(f"Schema creation (may already exist): {e}")

    def add_session(
        self,
        session_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Add or update a session node."""
        try:
            started_at = metadata.get("started_at", datetime.datetime.now())

            # Handle string timestamps
            if isinstance(started_at, str):
                try:
                    started_at = datetime.datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                except Exception:
                    started_at = datetime.datetime.now()

            if isinstance(started_at, (int, float)):
                started_at = datetime.datetime.fromtimestamp(started_at)

            summary = metadata.get("summary", "")[:500]

            query = f"""
                MERGE (s:Session {{id: '{session_id}'}})
                ON CREATE SET
                    s.source = '{metadata.get('source', 'unknown')}',
                    s.started_at = timestamp('{started_at.isoformat()}'),
                    s.message_count = {metadata.get('message_count', 0)},
                    s.summary = '{self._escape(summary)}'
            """

            self.conn.execute(query)
            return True

        except Exception as e:
            logger.error(f"Failed to add session {session_id}: {e}")
            return False

    def add_entity(
        self,
        session_id: str,
        entity: Dict[str, str]
    ) -> bool:
        """Add entity and link to session."""
        try:
            name = self._escape(entity["name"])
            entity_type = self._escape(entity.get("type", "unknown"))
            context = self._escape(entity.get("context", ""))

            query = f"""
                MERGE (e:Entity {{name: '{name}'}})
                ON CREATE SET e.type = '{entity_type}', e.context = '{context}'
            """
            self.conn.execute(query)

            query = f"""
                MATCH (s:Session {{id: '{session_id}'}}), (e:Entity {{name: '{name}'}})
                MERGE (s)-[:MENTIONS]->(e)
            """
            self.conn.execute(query)

            return True

        except Exception as e:
            logger.error(f"Failed to add entity: {e}")
            return False

    def add_topic(self, session_id: str, topic: str) -> bool:
        """Add topic and link to session."""
        try:
            topic_clean = self._escape(topic.lower())

            query = f"""
                MERGE (t:Topic {{name: '{topic_clean}'}})
                ON MATCH SET t.frequency = t.frequency + 1
            """
            self.conn.execute(query)

            query = f"""
                MATCH (s:Session {{id: '{session_id}'}}), (t:Topic {{name: '{topic_clean}'}})
                MERGE (s)-[:DISCUSSES]->(t)
            """
            self.conn.execute(query)

            return True

        except Exception as e:
            logger.error(f"Failed to add topic: {e}")
            return False

    def link_sessions(
        self,
        session1_id: str,
        session2_id: str,
        relation_type: str = "semantic",
        strength: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create relationship between sessions."""
        try:
            if relation_type == "correction":
                old_val = self._escape(metadata.get("old_value", "") if metadata else "")
                new_val = self._escape(metadata.get("new_value", "") if metadata else "")
                field = self._escape(metadata.get("field", "") if metadata else "")

                query = f"""
                    MATCH (s1:Session {{id: '{session1_id}'}}),
                          (s2:Session {{id: '{session2_id}'}})
                    CREATE (s2)-[:CORRECTS {{field: '{field}',
                                             old_value: '{old_val}',
                                             new_value: '{new_val}'}}]->(s1)
                """
            else:
                query = f"""
                    MATCH (s1:Session {{id: '{session1_id}'}}),
                          (s2:Session {{id: '{session2_id}'}})
                    MERGE (s1)-[r:RELATED_TO]->(s2)
                    ON CREATE SET r.strength = {strength}
                """

            self.conn.execute(query)
            return True

        except Exception as e:
            logger.error(f"Failed to link sessions: {e}")
            return False

    def find_related(
        self,
        session_ids: List[str],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find sessions related to given sessions."""
        try:
            if not session_ids:
                return []

            ids_str = ", ".join([f"'{sid}'" for sid in session_ids])

            query = f"""
                MATCH (s:Session)-[r:RELATED_TO|CORRECTS]->(related:Session)
                WHERE s.id IN [{ids_str}]
                RETURN related.id AS session_id,
                       CASE WHEN r.field IS NOT NULL THEN 'CORRECTS' ELSE 'RELATED_TO' END AS relation_type,
                       r.strength AS score
                LIMIT {limit}
            """

            result = self.conn.execute(query)

            related = []
            while result.has_next():
                row = result.get_next()
                related.append({
                    "session_id": row[0],
                    "relation_type": row[1],
                    "score": row[2] if row[2] else 0.5
                })

            return related

        except Exception as e:
            logger.error(f"Failed to find related: {e}")
            return []

    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get full context for a session."""
        try:
            context = {
                "session_id": session_id,
                "entities": [],
                "related_sessions": [],
                "corrected_by": [],
                "corrects": []
            }

            query = f"""
                MATCH (s:Session {{id: '{session_id}'}})-[:MENTIONS]->(e:Entity)
                RETURN e.name, e.type, e.context
            """
            result = self.conn.execute(query)
            while result.has_next():
                row = result.get_next()
                context["entities"].append({
                    "name": row[0],
                    "type": row[1],
                    "context": row[2]
                })

            query = f"""
                MATCH (s:Session {{id: '{session_id}'}})-[r:RELATED_TO]->(related:Session)
                RETURN related.id, r.strength
            """
            result = self.conn.execute(query)
            while result.has_next():
                row = result.get_next()
                context["related_sessions"].append({
                    "session_id": row[0],
                    "strength": row[1]
                })

            query = f"""
                MATCH (s:Session {{id: '{session_id}'}})-[c:CORRECTS]->(corrected:Session)
                RETURN corrected.id, c.field, c.old_value, c.new_value
            """
            result = self.conn.execute(query)
            while result.has_next():
                row = result.get_next()
                context["corrects"].append({
                    "session_id": row[0],
                    "field": row[1],
                    "old_value": row[2],
                    "new_value": row[3]
                })

            query = f"""
                MATCH (corrector:Session)-[c:CORRECTS]->(s:Session {{id: '{session_id}'}})
                RETURN corrector.id, c.field, c.old_value, c.new_value
            """
            result = self.conn.execute(query)
            while result.has_next():
                row = result.get_next()
                context["corrected_by"].append({
                    "session_id": row[0],
                    "field": row[1],
                    "old_value": row[2],
                    "new_value": row[3]
                })

            return context

        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return {"session_id": session_id, "error": str(e)}

    def query(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Execute raw Cypher query."""
        try:
            result = self.conn.execute(cypher_query)

            results = []
            column_names = result.get_column_names()

            while result.has_next():
                row = result.get_next()
                row_dict = {}
                for i, col in enumerate(column_names):
                    row_dict[col] = row[i]
                results.append(row_dict)

            return results

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    def get_stats(self) -> Dict[str, int]:
        """Get graph statistics."""
        try:
            stats = {}

            for table in ["Session", "Entity", "Topic"]:
                try:
                    result = self.conn.execute(f"MATCH (n:{table}) RETURN count(n)")
                    if result.has_next():
                        stats[table.lower()] = result.get_next()[0]
                except Exception:
                    stats[table.lower()] = 0

            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def _escape(self, s: str) -> str:
        """Escape string for Cypher queries."""
        if not s:
            return ""
        return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")[:1000]

    def close(self):
        """Close Kuzu connection."""
        if self.conn:
            pass
        if self.db:
            pass
