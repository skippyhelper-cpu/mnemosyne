"""
Mnemosyne - Local Semantic Memory Stack for Hermes

A multi-tier memory system combining:
- ChromaDB (embedded): Vector similarity search
- Kuzu (embedded): Knowledge graph relationships
- SQLite (existing): Verbatim message storage

Architecture:
    Tier 1: Working memory (hot cache with SHAKE-256 receipts)
    Tier 2: Semantic vectors (ChromaDB + query-frequency boosting)
    Tier 3: Knowledge graph (Kuzu Cypher queries, multi-hop traversal)
    Tier 4: Pattern synthesis (canonical entries from recurring topics)
"""

from .core.memory import MnemosyneMemory
from .core.config import MnemosyneConfig

__version__ = "1.1.0"
__all__ = ["MnemosyneMemory", "MnemosyneConfig"]
