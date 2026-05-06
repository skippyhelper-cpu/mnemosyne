"""Configuration for Mnemosyne memory system."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


def get_mnemosyne_home() -> Path:
    """Get the Mnemosyne data directory."""
    hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    return hermes_home / "mnemosyne"


@dataclass
class MnemosyneConfig:
    """Configuration for Mnemosyne memory stack."""

    # Data paths
    data_dir: Path = field(default_factory=get_mnemosyne_home)

    # Embedding model (nomic-embed-text-v1.5 = 768 dims, Apache 2.0)
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dims: int = 768

    # Kuzu graph settings
    kuzu_path: Optional[Path] = None

    # Indexing settings
    chunk_size: int = 512
    chunk_overlap: int = 128
    min_chunk_size: int = 100

    # Retrieval settings
    top_k_semantic: int = 5
    top_k_graph: int = 10
    similarity_threshold: float = 0.7

    # Cache settings
    cache_ttl_seconds: float = 3600

    def __post_init__(self):
        """Set up derived paths."""
        if self.kuzu_path is None:
            self.kuzu_path = self.data_dir / "graph.kuzu"

        # Ensure data dir exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
