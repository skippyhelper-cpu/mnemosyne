# Mnemosyne 🧠

**Local semantic memory stack for Hermes agents.**

Store, search, and reason over your agent's conversation history using vector similarity + knowledge graph + self-learning cache.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Mnemosyne Memory Stack                                 │
│                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐  │
│  │   sqlite-vec │   │    Kuzu     │   │   SQLite     │  │
│  │   (ANN + FTS│ + │ (Knowledge   │ + │  (relational │  │
│  │    hybrid)  │   │    Graph)    │   │   + FTS5)    │  │
│  └─────────────┘   └─────────────┘   └──────────────┘  │
│       vectors           graph            metadata        │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │           Self-Learning (SHAKE-256 cache)         │   │
│  │   query receipts → hot-vector boost + synthesis  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Three storage layers:**
- **sqlite-vec** — ANN vector search (k-NN) + FTS5 keyword search, embedded in SQLite, zero external services
- **Kuzu** — Property graph for entities, topics, session relationships, and correction tracking
- **SQLite + FTS5** — Relational metadata storage with full-text search

## Components

| Component | Purpose |
|-----------|---------|
| `sqlite-vec` | Vector embeddings (FastEmbed/BGE-M3) + ANN index + FTS5 keyword |
| `Kuzu` | Knowledge graph: entities, topics, session relationships, corrections |
| `SelfLearningManager` | Query cache (SHAKE-256), hot-vector session promotion, pattern synthesis |
| `MCP server` | Model Context Protocol tools for hermes-agent integration |
| `importers/` | Import from OpenClaw, Hermes SQLite, Notion, Google Drive |

## Quick Start

### Prebuilt Binary (fastest)

```bash
# Download from GitHub Releases
chmod +x mnemosyne
./mnemosyne stats
./mnemosyne search "what did we discuss about solar"
```

### Python Package

```bash
pip install mnemosyne-memory

# Set data directory
export MNEMOSYNE_HOME=~/.mnemosyne

mnemosyne stats
mnemosyne search "your query"
```

### From Source

```bash
git clone https://github.com/NousResearch/mnemosyne.git
cd mnemosyne
pip install -e ".[vec]"

export MNEMOSYNE_HOME=~/.mnemosyne
mnemosyne stats
```

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_HOME` | `~/.hermes/mnemosyne` | Data directory |
| `MNEMOSYNE_MODEL` | `BAAI/bge-m3` | Embedding model |
| `MNEMOSYNE_EMBED_DIM` | `1024` | Embedding dimension |
| `MNEMOSYNE_FTS_LANGUAGE` | `english` | FTS5 language |

## CLI Commands

```bash
mnemosyne search "query"              # Semantic + keyword search
mnemosyne stats                        # Show index sizes and stats
mnemosyne context <session_id>        # Get entities + relationships for a session
mnemosyne import --db <path>          # Import from Hermes SQLite
```

## MCP Tools (for hermes-agent)

When used as an MCP server:
- `mnemosyne__search` — Semantic search across sessions
- `mnemosyne__get_context` — Get entities + relationships for a session
- `mnemosyne__synthesize` — Trigger pattern synthesis
- `mnemosyne__cache_stats` — Show query cache stats

## Data Storage

- `~/.hermes/mnemosyne/mnemosyne.db` — SQLite with sqlite-vec + FTS5
- `~/.hermes/mnemosyne/graph.kuzu` — Kuzu graph database

## License

MIT
