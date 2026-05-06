"""Core Mnemosyne memory orchestrator."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .config import MnemosyneConfig
from .sqlite_vec_store import SqliteVecStore
from .kuzu_graph import KuzuGraph
from .self_learning import SelfLearningManager

logger = logging.getLogger(__name__)


class MnemosyneMemory:
    """
    Multi-tier memory system for Hermes.

    Combines:
    - SqliteVecStore (sqlite-vec + FTS5): Hybrid semantic + keyword search
    - Kuzu: Knowledge graph with Cypher queries and multi-hop traversal
    - SelfLearningManager: Three-tier learning (cache + hot-vectors + synthesis)
    """

    def __init__(self, config: Optional[MnemosyneConfig] = None):
        self.config = config or MnemosyneConfig()
        self.vector_store: Optional[SqliteVecStore] = None
        self.kuzu: Optional[KuzuGraph] = None
        self._initialized = False
        self.sl: Optional[SelfLearningManager] = None

    def initialize(self) -> bool:
        """Initialize all memory tiers. Returns True on success."""
        if self._initialized:
            return True

        try:
            logger.info("Initializing Mnemosyne memory...")

            # Initialize vector store (sqlite-vec + FTS5)
            self.vector_store = SqliteVecStore(config=self.config)
            if not self.vector_store.initialize():
                logger.error("Failed to initialize SqliteVecStore")
                return False

            # Initialize Kuzu (graph store)
            self.kuzu = KuzuGraph(self.config)
            if not self.kuzu.initialize():
                logger.error("Failed to initialize Kuzu")
                return False

            # Initialize self-learning manager
            self.sl = SelfLearningManager(config=self.config, auto_init=True)

            self._initialized = True
            logger.info("Mnemosyne initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Mnemosyne initialization failed: {e}")
            return False

    def index_session(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Index a session into vector and graph stores.

        Args:
            session_id: Unique session identifier
            messages: List of messages with role, content, timestamp
            metadata: Optional session metadata (source, model, etc.)

        Returns:
            True if indexed successfully
        """
        if not self._initialized and not self.initialize():
            return False

        try:
            session_text = self._extract_session_text(messages)
            if not session_text:
                logger.warning(f"No content to index for session {session_id}")
                return False

            # Index in vector store (sqlite-vec + FTS5)
            self.vector_store.index_session(
                session_id=session_id,
                text=session_text,
                metadata=metadata or {}
            )

            # Extract entities and add to graph
            entities = self._extract_entities(session_text, messages)
            self.kuzu.add_session(session_id, metadata or {})

            for entity in entities:
                self.kuzu.add_entity(session_id, entity)

            logger.debug(f"Indexed session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to index session {session_id}: {e}")
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
        Hybrid search across vector and graph stores.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters (source, date range, etc.)
            preferred_sources: Sources to boost in ranking (e.g. ['telegram', 'cli'])

        Returns:
            List of results with session_id, score, metadata
        """
        if not self._initialized and not self.initialize():
            return []

        results = []

        # Vector search via sqlite-vec
        try:
            semantic_results = self.vector_store.search(
                query,
                top_k=top_k,
                filters=filters,
                preferred_sources=preferred_sources,
                excluded_sources=excluded_sources,
            )
            results.extend(semantic_results)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")

        # Graph traversal for related sessions
        try:
            if results:
                session_ids = [r["session_id"] for r in results]
                related = self.kuzu.find_related(session_ids, limit=top_k)
                results.extend(related)
        except Exception as e:
            logger.error(f"Graph search failed: {e}")

        # Deduplicate and sort by score
        seen = set()
        unique_results = []
        for r in sorted(results, key=lambda x: x.get("score", 0), reverse=True):
            if r["session_id"] not in seen:
                seen.add(r["session_id"])
                unique_results.append(r)

        return unique_results[:top_k]

    def search_with_learning(
        self,
        query: str,
        session_key: str = "default",
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Hybrid search with full three-tier self-learning pipeline.

        Tier 1 — Query Cache: SHAKE-256 receipts, session-scoped invalidation, TTL.
        Tier 2 — Hot Vectors: query-frequency tracking, recency-biased score boosting.
        Tier 3 — Pattern Synthesis: canonical entries from recurring topics.
        """
        if not self._initialized and not self.initialize():
            return {"results": [], "source": "fresh", "cache_stats": {}, "boost_info": {}}

        if self.sl is None:
            logger.warning("SelfLearningManager not initialized")
            return {
                "results": self.search(query, top_k, filters),
                "source": "fresh",
                "cache_stats": {},
                "boost_info": {}
            }

        def base_search():
            return self.search(query, top_k=top_k * 2, filters=filters)

        results, source = self.sl.search_with_cache(
            query=query,
            session_key=session_key,
            search_fn=base_search,
            top_k=top_k,
            filters=filters,
        )

        boosted = self.sl.apply_boost(results, top_k=top_k)
        stats = self.sl.get_all_stats()

        return {
            "results": boosted,
            "source": source,
            "cache_stats": stats.get("query_cache", {}),
            "boost_info": {
                "hot_sessions": stats.get("hot_vectors", {}).get("hot_sessions", []),
            },
        }

    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get full context for a session including related sessions."""
        if not self._initialized:
            return None

        try:
            context = self.kuzu.get_session_context(session_id)

            if self.vector_store:
                neighbors = self.vector_store.get_neighbors(session_id)
                context["semantic_neighbors"] = neighbors

            return context
        except Exception as e:
            logger.error(f"Failed to get context for {session_id}: {e}")
            return None

    def query_graph(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query on the knowledge graph.

        Example:
            MATCH (s:Session)-[:MENTIONS]->(e:Entity {type: 'project'})
            RETURN s.id, e.name
        """
        if not self._initialized:
            return []

        return self.kuzu.query(cypher_query)

    def close(self):
        """Close all connections."""
        if self.vector_store:
            self.vector_store.close()
        if self.kuzu:
            self.kuzu.close()
        self._initialized = False

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _extract_session_text(self, messages: List[Dict[str, Any]]) -> str:
        """Extract searchable text from session messages."""
        texts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if content and role in ("user", "assistant"):
                texts.append(f"[{role}] {content}")
        return "\n\n".join(texts)

    def _extract_entities(
        self,
        session_text: str,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Extract entities from session for graph indexing.

        Simple rule-based extraction. Replace with NER for production.
        """
        entities = []
        text_lower = session_text.lower()

        # Common project patterns
        projects = [
            "solar", "finefighters", "hexrider", "wam", "motobikast",
            "filspire", "thinking rider", "solarforecast", "solargenius",
            "openclaw", "hermes", "mnemosyne"
        ]
        for project in projects:
            if project in text_lower:
                entities.append({
                    "name": project,
                    "type": "project",
                    "context": self._extract_context(text_lower, project)
                })

        # Person patterns
        for name in ("monika", "jessica", "jess", "filip"):
            if name in text_lower:
                entities.append({
                    "name": name,
                    "type": "person",
                    "context": "family" if name in ("monika", "jessica", "jess") else "user"
                })

        # Location patterns
        for loc in ("croydon", "bebington", "london"):
            if loc in text_lower:
                entities.append({
                    "name": loc,
                    "type": "location",
                    "context": "home"
                })

        # Technology patterns
        for tech in ("firebase", "docker", "sqlite", "postgres", "python",
                     "react", "fastapi", "kuzu", "chroma", "qdrant",
                     "telegram", "github"):
            if tech in text_lower:
                entities.append({
                    "name": tech,
                    "type": "technology",
                    "context": "stack"
                })

        return entities

    def _extract_context(self, text: str, entity: str, window: int = 100) -> str:
        """Extract surrounding context for an entity."""
        idx = text.find(entity)
        if idx == -1:
            return ""
        start = max(0, idx - window)
        end = min(len(text), idx + len(entity) + window)
        return text[start:end].replace("\n", " ")
