"""sqlite-vec + FTS5 vector store for semantic + keyword search."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import sqlite_vec

from .config import MnemosyneConfig

logger = logging.getLogger(__name__)


class SqliteVecStore:
    """
    Combined vector (sqlite-vec ANN) + full-text (FTS5) store.

    Schema
    ------
    vec_index(session_id, embedding)  — ANN via sqlite-vec
    fts_index(session_id, content)   — FTS5 via SQLite
    session_meta(session_id, text_preview, source, agent, started_at, metadata_json)
    """

    VEC_TABLE = "vec_index"
    FTS_TABLE = "fts_index"
    META_TABLE = "session_meta"

    def __init__(self, db_path: Optional[Path] = None, config: Optional[MnemosyneConfig] = None):
        if config:
            self.data_dir = config.data_dir
            self.db_path = self.data_dir / "mnemosyne.db"
        else:
            default_dir = Path.home() / ".hermes" / "mnemosyne"
            self.data_dir = db_path or default_dir
            self.db_path = self.data_dir / "mnemosyne.db"

        self.conn: Optional[Any] = None
        self._embedding_model: Any = None
        self._embedding_dim: int = 768
        self._init_model()

    def _init_model(self):
        """Load fastembed model."""
        try:
            from fastembed import TextEmbedding
            logger.info("Loading embedding model: nomic-ai/nomic-embed-text-v1.5")
            self._embedding_model = TextEmbedding("nomic-ai/nomic-embed-text-v1.5")
            # Get actual dimension from first embed
            sample = list(self._embedding_model.embed(["test"]))[0]
            self._embedding_dim = len(sample)
            logger.info(f"Embedding model loaded, dim={self._embedding_dim}")
        except ImportError:
            logger.warning("fastembed not installed, using pass-through embeddings")
            self._embedding_model = None

    def initialize(self) -> bool:
        """Open DB, load sqlite-vec, create schema."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self._create_tables()
            logger.info(f"SqliteVecStore initialized ✓ (db={self.db_path})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SqliteVecStore: {e}")
            return False

    def _create_tables(self):
        """Create virtual tables and metadata table if not exist."""
        self.conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.VEC_TABLE} USING vec0()")
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.FTS_TABLE} USING fts5(
                session_id UNINDEXED,
                content
            )
        """)
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.META_TABLE} (
                session_id TEXT PRIMARY KEY,
                text_preview TEXT,
                source TEXT,
                agent TEXT,
                started_at REAL,
                metadata_json TEXT
            )
        """)
        self.conn.commit()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts."""
        if self._embedding_model:
            return [list(v) for v in self._embedding_model.embed(texts)]
        # Fallback: random vectors
        import random
        return [[random.random() for _ in range(self._embedding_dim)] for _ in texts]

    def index_session(
        self,
        session_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Index a session: vector + FTS entry + metadata."""
        if not self.conn:
            return False
        try:
            emb = self._embed([content])[0]
            vec_bytes = sqlite_vec.serialize_float32(emb)

            self.conn.execute(
                f"INSERT OR REPLACE INTO {self.VEC_TABLE} (session_id, embedding) VALUES (?, ?)",
                (session_id, vec_bytes),
            )
            self.conn.execute(
                f"INSERT OR REPLACE INTO {self.FTS_TABLE} (session_id, content) VALUES (?, ?)",
                (session_id, content),
            )

            meta = metadata or {}
            self.conn.execute(
                f"INSERT OR REPLACE INTO {self.META_TABLE} VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    content[:200],
                    meta.get("source", ""),
                    meta.get("agent", ""),
                    meta.get("started_at"),
                    json.dumps(meta),
                ),
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to index session {session_id}: {e}")
            self.conn.rollback()
            return False

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        preferred_sources: Optional[List[str]] = None,
        excluded_sources: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: vector KNN with preferred-source boost.

        Two-pass: preferred sources returned first, then non-preferred fill.
        """
        if not self.conn:
            return []

        try:
            query_vec = self._embed([query])[0]
            query_json = json.dumps(query_vec)

            # Build base filter
            filter_parts: List[str] = []
            filter_params: List[Any] = []
            if filters:
                for key, val in filters.items():
                    if key == "source":
                        filter_parts.append("m.source = ?")
                        filter_params.append(val)
                    elif key == "agent":
                        filter_parts.append("m.agent = ?")
                        filter_params.append(val)

            # Exclude sources (e.g. cron noise)
            if excluded_sources:
                for src in excluded_sources:
                    filter_parts.append("m.source != ?")
                    filter_params.append(src)

            base_filter = (" AND " + " AND ".join(filter_parts)) if filter_parts else ""

            def run_knn(extra_filter: str, extra_params: List[Any], limit: int) -> List[Dict[str, Any]]:
                sql = f"""
                    SELECT
                        v.session_id,
                        distance,
                        m.text_preview,
                        m.source,
                        m.agent,
                        m.started_at,
                        m.metadata_json
                    FROM {self.VEC_TABLE} v
                    JOIN {self.META_TABLE} m ON v.session_id = m.session_id
                    WHERE v.embedding MATCH '{query_json}' AND k = {limit}
                    AND v.session_id NOT LIKE 'canonical:%%'
                    {base_filter}
                    {extra_filter}
                    ORDER BY distance
                """
                rows = self.conn.execute(sql, extra_params).fetchall()
                seen: Set[str] = set()
                results: List[Dict[str, Any]] = []
                for row in rows:
                    sid = row[0]
                    if sid in seen:
                        continue
                    seen.add(sid)
                    meta = {}
                    if row[6]:
                        try:
                            meta = json.loads(row[6])
                        except Exception:
                            pass
                    results.append({
                        "session_id": sid,
                        "score": 1.0 / (1.0 + float(row[1])),
                        "text_preview": row[2],
                        "metadata": {
                            "source": row[3],
                            "agent": row[4],
                            "started_at": row[5],
                            **meta,
                        }
                    })
                return results

            if preferred_sources:
                ph = ", ".join(["?" for _ in preferred_sources])
                pref_filter = f" AND m.source IN ({ph})"
                pref_params = list(filter_params) + preferred_sources
                pref_results = run_knn(pref_filter, pref_params, top_k)

                if len(pref_results) >= top_k:
                    return pref_results[:top_k]

                non_pref_filter = f" AND m.source NOT IN ({ph})"
                non_pref_params = list(filter_params) + preferred_sources
                remaining = run_knn(non_pref_filter, non_pref_params, top_k - len(pref_results))
                return pref_results + remaining
            else:
                return run_knn("", filter_params, top_k)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_fts_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Pure FTS5 keyword search."""
        if not self.conn:
            return []
        try:
            rows = self.conn.execute(
                f"""
                SELECT session_id, content, bm25 FROM {self.FTS_TABLE}
                WHERE {self.FTS_TABLE} MATCH ?
                ORDER BY bm25 LIMIT ?
                """,
                (query, top_k)
            ).fetchall()
            return [
                {
                    "session_id": row[0],
                    "score": abs(row[2]) if row[2] else 0.0,
                    "text_preview": (row[1] or "")[:200],
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"FTS search failed: {e}")
            return []

    def get_neighbors(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find semantically similar sessions by fetching this session's vector and doing KNN."""
        if not self.conn:
            return []
        try:
            row = self.conn.execute(
                f"SELECT embedding FROM {self.VEC_TABLE} WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            if not row:
                return []
            query_vec = list(sqlite_vec.deserialize_float32(row[0]))
            query_json = json.dumps(query_vec)

            rows = self.conn.execute(
                f"""
                SELECT session_id, distance
                FROM {self.VEC_TABLE}
                WHERE session_id != ?
                AND embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                (session_id, query_json, limit)
            ).fetchall()
            return [
                {
                    "session_id": r[0],
                    "score": 1.0 / (1.0 + float(r[1])),
                    "relation_type": "semantic_similarity",
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Failed to get neighbors: {e}")
            return []

    def delete_session(self, session_id: str) -> bool:
        """Remove a session from all stores."""
        if not self.conn:
            return False
        try:
            self.conn.execute(f"DELETE FROM {self.VEC_TABLE} WHERE session_id = ?", (session_id,))
            self.conn.execute(f"DELETE FROM {self.FTS_TABLE} WHERE session_id = ?", (session_id,))
            self.conn.execute(f"DELETE FROM {self.META_TABLE} WHERE session_id = ?", (session_id,))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        if not self.conn:
            return {"status": "not_initialized"}
        try:
            vec_count = self.conn.execute(f"SELECT COUNT(*) FROM {self.VEC_TABLE}").fetchone()[0]
            fts_count = self.conn.execute(f"SELECT COUNT(*) FROM {self.FTS_TABLE}").fetchone()[0]
            meta_count = self.conn.execute(f"SELECT COUNT(*) FROM {self.META_TABLE}").fetchone()[0]
            return {
                "vectors_count": vec_count,
                "fts_entries": fts_count,
                "metadata_count": meta_count,
                "status": "ready",
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


# Avoid top-level import conflict — sqlite3 loaded lazily inside methods
import sqlite3
