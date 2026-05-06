# Mnemosyne 🧠

**Local semantic memory stack for Hermes agents.**

Store, search, and reason over your agent's conversation history using vector similarity + knowledge graph + self-learning cache.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Mnemosyne Memory Stack                                 │
│                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐  │
│  │  sqlite-vec │ + │    Kuzu     │ + │   SQLite     │  │
│  │  (ANN + FTS│   │ (Knowledge   │   │  (relational │  │
│  │   hybrid)   │   │    Graph)    │   │  + FTS5)    │  │
│  └─────────────┘   └─────────────┘   └──────────────┘  │
│       vectors           graph            metadata        │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │        Self-Learning (SHAKE-256 cache)           │   │
│  │  query receipts → hot-vector boost + synthesis   │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Three storage layers:**
- **sqlite-vec** — ANN vector search (k-NN) + FTS5 keyword search, embedded in SQLite, zero external services
- **Kuzu** — Property graph for entities, topics, session relationships, and correction tracking
- **SQLite + FTS5** — Relational metadata storage with full-text search

## Features

- **Auto-setup** — runs automatically on first use, no manual configuration
- **Incremental sync** — new sessions indexed every 30 min via hermes-agent cron job
- **Privacy controls** — opt in/out of each data source via config file
- **Hybrid search** — combines semantic (vector) + keyword (FTS5) for accurate results
- **Knowledge graph** — entity extraction, topic tracking, correction tracking
- **Self-learning** — SHAKE-256 query cache, hot-vector session boosting

## Quick Start

```bash
pip install mnemosyne-memory[vec]

# First run — auto-setup runs automatically (indexes all sessions)
mnemosyne search "what did we discuss about solar"

# Check sync status
mnemosyne sync --check

# See statistics
mnemosyne stats
```

That's it. Setup is automatic. New sessions are picked up every 30 minutes.

## Data Sources

Mnemosyne can sync from multiple sources. By default it auto-detects what's installed:

| Source | Default | Config key |
|--------|---------|------------|
| Hermes-agent sessions (`~/.hermes/state.db`) | **On** | `hermes` |
| OpenClaw sessions (`~/.openclaw/agents/*/sessions/`) | Auto (on if dir exists) | `openclaw` |

### Privacy Controls

Edit `~/.config/mnemosyne/sources.toml` to enable/disable sources:

```toml
[sources]
openclaw = false   # disable if not installed or not wanted
hermes = true      # default on
```

Mnemosyne creates this file automatically on first run.

## CLI Commands

```bash
mnemosyne search "query"              # Semantic + keyword search
mnemosyne stats                        # Show index sizes and stats
mnemosyne context <session_id>         # Get entities + relationships
mnemosyne sync                         # Run incremental sync now
mnemosyne sync --check                 # Show sync status (no changes)
mnemosyne sync --source hermes         # Sync only hermes
mnemosyne setup                        # Re-run setup (sync + cron registration)
```

## MCP Tools (for hermes-agent)

When connected as an MCP server, four tools are available:

- `mnemosyne__search` — Hybrid semantic + keyword search across sessions
- `mnemosyne__get_context` — Get entities + relationships for a session
- `mnemosyne__get_related` — Find related sessions via knowledge graph
- `mnemosyne__stats` — Show index sizes and cache statistics

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_HOME` | `~/.hermes/mnemosyne` | Data directory |

Most settings are auto-configured. To override embedding model or dimension, see `mnemosyne/core/config.py`.

## Data Storage

```
~/.hermes/mnemosyne/
├── mnemosyne.db         # SQLite with sqlite-vec + FTS5
└── graph.kuzu           # Kuzu graph database

~/.config/mnemosyne/
└── sources.toml         # Data source configuration
```

## License

MIT
