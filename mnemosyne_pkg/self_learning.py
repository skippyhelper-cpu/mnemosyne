"""Mnemosyne Self-Learning Layer.

Three-tier learning system:
- Tier 1 (Instant): Query result caching with SHAKE-256 receipts
- Tier 2 (Background): Hot vector promotion based on query frequency
- Tier 3 (Deep): Pattern synthesis — canonical entry creation from recurring topics
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import MnemosyneConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tier 1: Query Cache
# --------------------------------------------------------------------------- #

@dataclass
class CacheEntry:
    query_hash: str
    query_text: str
    session_key: str
    results_json: str
    receipt: str
    created_at: float
    last_hit: float
    hit_count: int = 0
    source: str = "cache"


@dataclass
class CacheStats:
    entries: int
    hits: int
    misses: int
    hit_rate: float
    storage_bytes: int


class QueryCache:
    """
    Tier 1 self-learning: instant query result caching with SHAKE-256 receipts.
    """

    def __init__(self, kuzu_conn):
        self.conn = kuzu_conn
        self._stats_lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0}

    def initialize(self) -> bool:
        try:
            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS QueryCache(
                    query_hash STRING PRIMARY KEY,
                    query_text STRING,
                    session_key STRING,
                    results_json STRING,
                    receipt STRING,
                    created_at DOUBLE,
                    last_hit DOUBLE,
                    hit_count INT64 DEFAULT 0,
                    source STRING DEFAULT 'cache'
                )
            """)
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS CACHED(
                    FROM Session TO QueryCache,
                    MANY_MANY
                )
            """)
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS NEAR_CACHE(
                    FROM QueryCache TO QueryCache,
                    similarity DOUBLE,
                    MANY_MANY
                )
            """)
            logger.info("QueryCache schema initialized")
            return True
        except Exception as e:
            logger.error(f"QueryCache init failed: {e}")
            return False

    def get(self, query: str, session_key: str) -> Tuple[Optional[List[Dict]], str]:
        query_hash = self._hash_query(query)
        esc_sk = self._escape(session_key)
        esc_qh = self._escape(query_hash)

        try:
            q = (
                "MATCH (s:Session {id: '%s'})-[:CACHED]->(c:QueryCache {query_hash: '%s'})"
                " RETURN c.query_hash, c.results_json, c.receipt, c.hit_count"
            ) % (esc_sk, esc_qh)
            result = self.conn.execute(q)

            if not result.has_next():
                with self._stats_lock:
                    self._stats["misses"] += 1
                return (None, "miss")

            row = result.get_next()
            stored_hash, results_json, stored_receipt, hit_count = row

            expected_receipt = self._make_receipt(stored_hash, results_json)
            if expected_receipt != stored_receipt:
                logger.warning("Cache receipt mismatch for %s... — evicting", query_hash[:16])
                self._evict(query_hash)
                with self._stats_lock:
                    self._stats["misses"] += 1
                return (None, "evicted")

            now = time.time()
            up_q = (
                "MATCH (c:QueryCache {query_hash: '%s'})"
                " SET c.hit_count = %d, c.last_hit = %s"
            ) % (esc_qh, hit_count + 1, now)
            self.conn.execute(up_q)

            with self._stats_lock:
                self._stats["hits"] += 1

            results = json.loads(results_json)
            return (results, "cache")

        except Exception as e:
            logger.error("Cache get failed: %s", e)
            with self._stats_lock:
                self._stats["misses"] += 1
            return (None, "miss")

    def put(
        self,
        query: str,
        session_key: str,
        results: List[Dict],
        source: str = "fresh"
    ) -> Optional[str]:
        query_hash = self._hash_query(query)
        esc_sk = self._escape(session_key)
        esc_qh = self._escape(query_hash)

        try:
            results_json = json.dumps(results, sort_keys=True)
            receipt = self._make_receipt(query_hash, results_json)
            now = time.time()
            esc_rj = self._escape(results_json)
            esc_qt = self._escape(query[:500])

            merge_q = (
                "MERGE (c:QueryCache {query_hash: '%s'}) "
                "ON CREATE SET c.query_text = '%s', c.session_key = '%s', "
                "c.results_json = '%s', c.receipt = '%s', c.created_at = %s, "
                "c.last_hit = %s, c.hit_count = 0, c.source = '%s' "
                "ON MATCH SET c.results_json = '%s', c.receipt = '%s', "
                "c.last_hit = %s, c.source = '%s'"
            ) % (esc_qh, esc_qt, esc_sk, esc_rj, receipt, now, now, source,
                 esc_rj, receipt, now, source)
            self.conn.execute(merge_q)

            link_q = (
                "MATCH (s:Session {id: '%s'}) "
                "MATCH (c:QueryCache {query_hash: '%s'}) "
                "MERGE (s)-[:CACHED]->(c)"
            ) % (esc_sk, esc_qh)
            self.conn.execute(link_q)

            return receipt

        except Exception as e:
            logger.error("Cache put failed: %s", e)
            return None

    def invalidate_session(self, session_key: str) -> int:
        try:
            esc_sk = self._escape(session_key)
            q = (
                "MATCH (s:Session {id: '%s'})-[:CACHED]->(c:QueryCache) "
                "DETACH DELETE c RETURN count(c)"
            ) % esc_sk
            result = self.conn.execute(q)
            count = result.get_next()[0] if result.has_next() else 0
            logger.info("Invalidated %d cache entries for session %s", count, session_key)
            return count
        except Exception as e:
            logger.error("Cache invalidation failed: %s", e)
            return 0

    def invalidate_expired(self, max_age_seconds: float = 3600) -> int:
        try:
            cutoff = time.time() - max_age_seconds
            q = (
                "MATCH (c:QueryCache) WHERE c.created_at < %s "
                "DETACH DELETE c RETURN count(c)"
            ) % cutoff
            result = self.conn.execute(q)
            count = result.get_next()[0] if result.has_next() else 0
            if count > 0:
                logger.info("Evicted %d expired cache entries", count)
            return count
        except Exception as e:
            logger.error("Cache expiry cleanup failed: %s", e)
            return 0

    def get_stats(self) -> CacheStats:
        try:
            result = self.conn.execute(
                "MATCH (c:QueryCache) RETURN count(c), sum(c.hit_count)"
            )
            entries = 0
            hits = 0
            if result.has_next():
                row = result.get_next()
                entries = row[0] or 0
                hits = row[1] or 0

            with self._stats_lock:
                misses = self._stats["misses"]
                total = self._stats["hits"] + misses
                hit_rate = self._stats["hits"] / total if total > 0 else 0.0

            return CacheStats(
                entries=entries,
                hits=hits,
                misses=misses,
                hit_rate=hit_rate,
                storage_bytes=entries * 2000
            )
        except Exception as e:
            logger.error("Failed to get cache stats: %s", e)
            return CacheStats(0, 0, 0, 0.0, 0)

    def _hash_query(self, query: str) -> str:
        normalized = " ".join(query.lower().split())
        return hashlib.shake_256(normalized.encode()).hexdigest(32)

    def _make_receipt(self, query_hash: str, results_json: str) -> str:
        return hashlib.shake_256(
            (query_hash + results_json).encode()
        ).hexdigest(32)

    def _evict(self, query_hash: str):
        try:
            q = "MATCH (c:QueryCache {query_hash: '%s'}) DETACH DELETE c"
            self.conn.execute(q % self._escape(query_hash))
        except Exception as e:
            logger.error("Evict failed: %s", e)

    def _escape(self, s: str) -> str:
        if not s:
            return ""
        return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")[:2000]


# --------------------------------------------------------------------------- #
# Tier 2: Hot Vector Tracking
# --------------------------------------------------------------------------- #

class HotVectorTracker:
    """
    Tier 2 self-learning: track query frequency and promote hot vectors.
    Boost score = query_count / (age_hours + 1)^0.3 (recency-biased).
    """

    def __init__(self, kuzu_conn):
        self.conn = kuzu_conn

    def initialize(self) -> bool:
        try:
            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS QueryFrequency(
                    session_id STRING PRIMARY KEY,
                    query_count INT64 DEFAULT 0,
                    last_queried DOUBLE DEFAULT 0,
                    avg_position DOUBLE DEFAULT 0.0,
                    boost_score DOUBLE DEFAULT 0.0
                )
            """)
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS TRACKS(
                    FROM QueryFrequency TO Session,
                    MANY_MANY
                )
            """)
            logger.info("HotVectorTracker schema initialized")
            return True
        except Exception as e:
            logger.error("HotVectorTracker init failed: %s", e)
            return False

    def record_result(self, session_id: str, position: int) -> bool:
        try:
            now = time.time()
            esc_sid = self._escape(session_id)

            merge_q = (
                "MERGE (f:QueryFrequency {session_id: '%s'}) "
                "ON CREATE SET f.query_count = 1, f.last_queried = %s, "
                "f.avg_position = %s, f.boost_score = 1.0 "
                "ON MATCH SET f.query_count = f.query_count + 1, "
                "f.last_queried = %s, "
                "f.avg_position = (f.avg_position * f.query_count + %s) / (f.query_count + 1)"
            ) % (esc_sid, now, float(position), now, float(position))
            self.conn.execute(merge_q)

            self._recompute_boost(session_id)
            return True

        except Exception as e:
            logger.error("Failed to record result: %s", e)
            return False

    def get_boost_scores(self, session_ids: List[str]) -> Dict[str, float]:
        if not session_ids:
            return {}

        try:
            ids_str = ", ".join(["'%s'" % self._escape(s) for s in session_ids])
            q = (
                "MATCH (f:QueryFrequency) WHERE f.session_id IN [%s] "
                "RETURN f.session_id, f.boost_score"
            ) % ids_str
            result = self.conn.execute(q)

            boosts = {}
            while result.has_next():
                row = result.get_next()
                boosts[row[0]] = row[1] if row[1] else 1.0

            for sid in session_ids:
                if sid not in boosts:
                    boosts[sid] = 1.0

            return boosts

        except Exception as e:
            logger.error("Failed to get boost scores: %s", e)
            return {sid: 1.0 for sid in session_ids}

    def get_hot_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            q = (
                "MATCH (f:QueryFrequency) WHERE f.query_count >= 3 "
                "RETURN f.session_id, f.query_count, f.last_queried, f.boost_score "
                "ORDER BY f.boost_score DESC LIMIT %d"
            ) % limit
            result = self.conn.execute(q)

            hot = []
            while result.has_next():
                row = result.get_next()
                hot.append({
                    "session_id": row[0],
                    "query_count": row[1],
                    "last_queried": row[2],
                    "boost_score": row[3]
                })
            return hot
        except Exception as e:
            logger.error("Failed to get hot sessions: %s", e)
            return []

    def _recompute_boost(self, session_id: str):
        try:
            esc_sid = self._escape(session_id)
            q = (
                "MATCH (f:QueryFrequency {session_id: '%s'}) "
                "RETURN f.query_count, f.last_queried"
            ) % esc_sid
            result = self.conn.execute(q)

            if not result.has_next():
                return

            row = result.get_next()
            query_count = row[0]
            last_queried = row[1]

            age_hours = (time.time() - last_queried) / 3600
            boost = query_count / ((age_hours + 1) ** 0.3)

            up_q = "MATCH (f:QueryFrequency {session_id: '%s'}) SET f.boost_score = %s"
            self.conn.execute(up_q % (esc_sid, boost))

        except Exception as e:
            logger.error("Boost recompute failed: %s", e)

    def decay_old_entries(self, max_age_days: float = 14) -> int:
        try:
            cutoff = time.time() - (max_age_days * 86400)
            q = (
                "MATCH (f:QueryFrequency) WHERE f.last_queried < %s "
                "SET f.boost_score = 1.0, f.query_count = 0 RETURN count(f)"
            ) % cutoff
            result = self.conn.execute(q)
            count = result.get_next()[0] if result.has_next() else 0
            if count > 0:
                logger.info("Decayed %d stale hot-vector entries", count)
            return count
        except Exception as e:
            logger.error("HotVector decay failed: %s", e)
            return 0

    def _escape(self, s: str) -> str:
        if not s:
            return ""
        return s.replace("\\", "\\\\").replace("'", "\\'")[:200]


# --------------------------------------------------------------------------- #
# Tier 3: Pattern Synthesis
# --------------------------------------------------------------------------- #

@dataclass
class CanonicalEntry:
    topic: str
    content: str
    source_sessions: List[str]
    confidence: float
    created_at: float


class PatternSynthesizer:
    """
    Tier 3 self-learning: deep pattern synthesis.

    Weekly job that:
    1. Scans Topic nodes in Kuzu for recurring themes
    2. Groups related sessions by topic
    3. Creates canonical entries — concise summaries of recurring topics
    """

    def __init__(self, kuzu_conn, vector_store=None):
        self.conn = kuzu_conn
        self.vector_store = vector_store

    def initialize(self) -> bool:
        try:
            self.conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS CanonicalEntry(
                    topic STRING PRIMARY KEY,
                    content STRING,
                    source_sessions_json STRING,
                    confidence DOUBLE,
                    created_at DOUBLE,
                    updated_at DOUBLE
                )
            """)
            self.conn.execute("""
                CREATE REL TABLE IF NOT EXISTS SYNTHESIZED_FROM(
                    FROM CanonicalEntry TO Session,
                    MANY_MANY
                )
            """)
            logger.info("PatternSynthesizer schema initialized")
            return True
        except Exception as e:
            logger.error("PatternSynthesizer init failed: %s", e)
            return False

    def synthesize(
        self,
        min_topic_frequency: int = 3,
        min_session_count: int = 3
    ) -> List[CanonicalEntry]:
        try:
            topics = self._find_recurring_topics(min_topic_frequency, min_session_count)
            created = []

            for topic_name, session_ids, session_texts in topics:
                entry = self._synthesize_entry(topic_name, session_ids, session_texts)
                if entry:
                    created.append(entry)

            logger.info("Synthesis complete: %d canonical entries", len(created))
            return created

        except Exception as e:
            logger.error("Synthesis failed: %s", e)
            return []

    def get_canonical(self, topic: str) -> Optional[CanonicalEntry]:
        try:
            esc_t = self._escape(topic)
            q = (
                "MATCH (c:CanonicalEntry {topic: '%s'}) "
                "RETURN c.topic, c.content, c.source_sessions_json, "
                "c.confidence, c.created_at"
            ) % esc_t
            result = self.conn.execute(q)

            if not result.has_next():
                return None

            row = result.get_next()
            return CanonicalEntry(
                topic=row[0],
                content=row[1],
                source_sessions=json.loads(row[2]),
                confidence=row[3],
                created_at=row[4]
            )
        except Exception as e:
            logger.error("Failed to get canonical: %s", e)
            return None

    def get_all_canonical(self) -> List[CanonicalEntry]:
        try:
            q = (
                "MATCH (c:CanonicalEntry) "
                "RETURN c.topic, c.content, c.source_sessions_json, "
                "c.confidence, c.created_at ORDER BY c.confidence DESC"
            )
            result = self.conn.execute(q)

            entries = []
            while result.has_next():
                row = result.get_next()
                entries.append(CanonicalEntry(
                    topic=row[0],
                    content=row[1],
                    source_sessions=json.loads(row[2]),
                    confidence=row[3],
                    created_at=row[4]
                ))
            return entries
        except Exception as e:
            logger.error("Failed to get all canonical: %s", e)
            return []

    def _find_recurring_topics(
        self,
        min_freq: int,
        min_session_count: int
    ) -> List[Tuple[str, List[str], List[str]]]:
        try:
            q = (
                "MATCH (t:Topic) WHERE t.frequency >= %d "
                "MATCH (s:Session)-[:DISCUSSES]->(t) "
                "WITH t.name as topic, collect(distinct s.id) as session_ids, "
                "count(distinct s.id) as num_sessions "
                "RETURN topic, session_ids, num_sessions "
                "ORDER BY num_sessions DESC LIMIT 20"
            ) % min_freq
            result = self.conn.execute(q)

            topics = []
            while result.has_next():
                row = result.get_next()
                topic_name, session_ids, num_sessions = row[0], row[1], row[2]
                if num_sessions < min_session_count:
                    continue
                session_texts = self._get_session_texts(session_ids)
                topics.append((topic_name, session_ids, session_texts))

            return topics

        except Exception as e:
            logger.error("Topic finding failed: %s", e)
            return []

    def _get_session_texts(self, session_ids: List[str]) -> List[str]:
        """Fetch text content for sessions from vector store or Kuzu."""
        texts = []
        for sid in session_ids[:10]:
            try:
                # Try vector store first (FTS5 text)
                if self.vector_store and self.vector_store.conn:
                    fts_row = self.vector_store.conn.execute(
                        "SELECT content FROM fts_index WHERE session_id = ?",
                        (sid,)
                    ).fetchone()
                    if fts_row and fts_row[0]:
                        texts.append(fts_row[0][:1000])
                        continue

                # Fall back to Kuzu session summary
                q = "MATCH (s:Session {id: '%s'}) RETURN s.summary"
                res = self.conn.execute(q % self._escape(sid))
                if res.has_next():
                    summary = res.get_next()[0]
                    if summary:
                        texts.append(summary[:1000])
            except Exception:
                continue
        return texts

    def _synthesize_entry(
        self,
        topic: str,
        session_ids: List[str],
        session_texts: List[str]
    ) -> Optional[CanonicalEntry]:
        if not session_texts:
            return None

        topic_lower = topic.lower()
        all_sentences = []

        for text in session_texts:
            sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".")]
            for s in sentences:
                if len(s) > 20 and topic_lower in s.lower():
                    all_sentences.append(s)

        if not all_sentences:
            content = session_texts[0][:500] if session_texts else ""
        else:
            seen = set()
            unique = []
            for s in all_sentences:
                key = s[:50].lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(s)
            content = " | ".join(unique[:5])[:1000]

        confidence = min(1.0, len(session_ids) / 10.0)
        now = time.time()
        sessions_json = json.dumps(session_ids[:50])
        esc_topic = self._escape(topic)
        esc_content = self._escape(content)

        try:
            merge_q = (
                "MERGE (c:CanonicalEntry {topic: '%s'}) "
                "ON CREATE SET c.content = '%s', c.source_sessions_json = '%s', "
                "c.confidence = %s, c.created_at = %s, c.updated_at = %s "
                "ON MATCH SET c.content = '%s', c.source_sessions_json = '%s', "
                "c.confidence = %s, c.updated_at = %s"
            ) % (esc_topic, esc_content, sessions_json, confidence, now, now,
                 esc_content, sessions_json, confidence, now)
            self.conn.execute(merge_q)

            for sid in session_ids[:20]:
                link_q = (
                    "MATCH (c:CanonicalEntry {topic: '%s'}) "
                    "MATCH (s:Session {id: '%s'}) "
                    "MERGE (c)-[:SYNTHESIZED_FROM]->(s)"
                )
                self.conn.execute(link_q % (esc_topic, self._escape(sid)))

            # Index canonical in vector store if available
            if self.vector_store and content:
                try:
                    self.vector_store.index_session(
                        session_id="canonical:%s" % topic,
                        text="Topic: %s\n\n%s" % (topic, content),
                        metadata={"type": "canonical", "topic": topic, "confidence": confidence}
                    )
                except Exception as e:
                    logger.warning("Failed to index canonical in vector store: %s", e)

            return CanonicalEntry(
                topic=topic,
                content=content,
                source_sessions=session_ids[:50],
                confidence=confidence,
                created_at=now
            )

        except Exception as e:
            logger.error("Failed to create canonical entry for %s: %s", topic, e)
            return None

    def _escape(self, s: str) -> str:
        if not s:
            return ""
        return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")[:2000]


# --------------------------------------------------------------------------- #
# SelfLearningManager — orchestrates all three tiers
# --------------------------------------------------------------------------- #

class SelfLearningManager:
    """
    Unified self-learning manager.

    Usage:
        sl = SelfLearningManager(config)
        sl.initialize()

        # On search — check cache first
        result = sl.search_with_cache(query, session_key, filters)

        # After search — record frequency
        sl.record_search_results(session_ids_with_positions)

        # Nightly — run synthesis
        sl.synthesize_patterns()
    """

    def __init__(self, config: Optional[MnemosyneConfig] = None, auto_init: bool = False):
        self.config = config or MnemosyneConfig()
        self._initialized = False
        self._kuzu_conn = None
        self._vector_store = None
        self.query_cache: Optional[QueryCache] = None
        self.hot_vectors: Optional[HotVectorTracker] = None
        self.synthesizer: Optional[PatternSynthesizer] = None

        if auto_init:
            self.initialize()

    def initialize(self) -> bool:
        """Initialize all three tiers."""
        if self._initialized:
            return True

        try:
            import kuzu
            logger.info("Initializing SelfLearningManager...")

            db = kuzu.Database(str(self.config.kuzu_path))
            conn = kuzu.Connection(db)
            self._kuzu_conn = conn

            # Initialize vector store (sqlite-vec) if available
            try:
                from .sqlite_vec_store import SqliteVecStore
                self._vector_store = SqliteVecStore(config=self.config)
                if self._vector_store.initialize():
                    logger.info("SelfLearning: SqliteVecStore ready")
                else:
                    logger.warning("SelfLearning: SqliteVecStore failed, disabling")
                    self._vector_store = None
            except Exception as e:
                logger.warning(f"SelfLearning: SqliteVecStore unavailable ({e}), disabling")
                self._vector_store = None

            # Initialize three tiers
            self.query_cache = QueryCache(conn)
            self.query_cache.initialize()

            self.hot_vectors = HotVectorTracker(conn)
            self.hot_vectors.initialize()

            self.synthesizer = PatternSynthesizer(conn, self._vector_store)
            self.synthesizer.initialize()

            self._initialized = True
            logger.info("SelfLearningManager initialized successfully")
            return True

        except Exception as e:
            logger.error("SelfLearningManager init failed: %s", e)
            return False

    def search_with_cache(
        self,
        query: str,
        session_key: str,
        search_fn,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        cache_ttl: float = 3600
    ) -> Tuple[List[Dict], str]:
        """
        Search with Tier 1 caching.

        1. Check cache for query+session
        2. If hit and valid -> return cached results
        3. If miss -> run search_fn(), cache results, return fresh
        """
        if not self._initialized:
            return search_fn(), "fresh"

        cached = self.query_cache.get(query, session_key)
        if cached[0] is not None:
            results, source = cached
            logger.debug("Cache HIT for query '%s...' [%s]", query[:50], source)
            return results, source

        # Cache miss — run fresh search
        results = search_fn()
        self.query_cache.put(query, session_key, results)

        for i, r in enumerate(results[:top_k]):
            sid = r.get("session_id")
            if sid:
                self.hot_vectors.record_result(sid, i)

        return results, "fresh"

    def record_search_results(self, results: List[Dict], top_k: int = 5):
        """Record search results for hot-vector tracking."""
        if not self._initialized:
            return
        for i, r in enumerate(results[:top_k]):
            sid = r.get("session_id")
            if sid:
                self.hot_vectors.record_result(sid, i)

    def apply_boost(
        self,
        results: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Apply hot-vector boost scores to search results.
        Multiplies score by boost^0.2 to promote hot sessions without
        drowning semantic relevance.
        """
        if not self._initialized or not results:
            return results

        session_ids = [r.get("session_id") for r in results if r.get("session_id")]
        boosts = self.hot_vectors.get_boost_scores(session_ids)

        boosted = []
        for r in results:
            sid = r.get("session_id")
            boost = boosts.get(sid, 1.0) if sid else 1.0
            new_score = min(1.0, r.get("score", 0.5) * (boost ** 0.2))
            r = dict(r)
            r["score"] = new_score
            r["boost"] = boost
            boosted.append(r)

        return sorted(boosted, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    def synthesize_patterns(
        self,
        min_topic_freq: int = 3,
        min_sessions: int = 3
    ) -> List[CanonicalEntry]:
        """Run Tier 3 pattern synthesis. Call periodically (e.g., weekly)."""
        if not self._initialized:
            return []
        return self.synthesizer.synthesize(min_topic_freq, min_sessions)

    def cleanup_expired_cache(self, max_age: float = 3600) -> int:
        """Remove expired cache entries. Call periodically."""
        if not self._initialized:
            return 0
        return self.query_cache.invalidate_expired(max_age)

    def decay_hot_vectors(self, max_age_days: float = 14) -> int:
        """Reset stale hot-vector entries."""
        if not self._initialized:
            return 0
        return self.hot_vectors.decay_old_entries(max_age_days)

    def get_all_stats(self) -> Dict[str, Any]:
        """Return stats from all three tiers."""
        if not self._initialized:
            return {"initialized": False}

        cache_stats = self.query_cache.get_stats()
        hot_sessions = self.hot_vectors.get_hot_sessions(limit=10)
        canonical = self.synthesizer.get_all_canonical()

        return {
            "initialized": True,
            "query_cache": {
                "entries": cache_stats.entries,
                "hits": cache_stats.hits,
                "misses": cache_stats.misses,
                "hit_rate": "%.1f%%" % (cache_stats.hit_rate * 100),
                "storage_mb": "%.2f" % (cache_stats.storage_bytes / 1e6),
            },
            "hot_vectors": {
                "tracked_sessions": len(hot_sessions),
                "top_5": hot_sessions[:5],
            },
            "canonical_entries": {
                "count": len(canonical),
                "topics": [c.topic for c in canonical],
            }
        }
