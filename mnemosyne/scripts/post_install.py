"""
Post-install hook for mnemosyne-memory.

Automatically called by pip after installation (via pyproject.toml
entry-points or setup.py script hook).

Performs:
  1. Full sync of all existing sessions (openclaw + hermes)
  2. Registers a cron job with hermes-agent for incremental syncs

Will skip gracefully if hermes-agent is not installed.
"""

import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("post_install")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _find_hermes_cron():
    """Find the hermes CLI tool for managing cron jobs."""
    candidates = [
        Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "hermes",
        Path.home() / ".local" / "bin" / "hermes",
        shutil.which("hermes"),
    ]
    for path in candidates:
        if path and path.exists():
            return str(path)
    return None


def _find_mnemosyne_python():
    """Find the Python interpreter that has mnemosyne installed."""
    candidates = [
        Path(sys.executable),  # Current Python
        Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python3.11",
        Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python3",
        shutil.py3,
        shutil.which("python3"),
    ]
    for path in candidates:
        if path and path.exists():
            return str(path)
    return sys.executable


def run_initial_sync(python_path: str):
    """Run a full sync of all session sources."""
    sync_script = Path(__file__).parent / "sync.py"
    if not sync_script.exists():
        logger.warning(f"Sync script not found at {sync_script}, skipping initial sync")
        return

    logger.info("Running initial full sync of all session sources...")
    t0 = time.time()

    for source in ("openclaw", "hermes"):
        logger.info(f"  Syncing {source}...")
        result = subprocess.run(
            [python_path, str(sync_script), "--source", source, "--verbose"],
            capture_output=False,
        )
        if result.returncode != 0:
            logger.warning(f"  {source} sync returned exit code {result.returncode}")

    elapsed = time.time() - t0
    logger.info(f"Initial sync complete in {elapsed:.1f}s")


def register_cron_job(python_path: str, sync_script: Path, interval_minutes: int = 30):
    """
    Register a hermes-agent cron job to run mnemosyne sync every N minutes.
    """
    hermes = _find_hermes_cron()
    if hermes is None:
        logger.info("hermes-agent not found — skipping cron registration")
        logger.info("After installing hermes-agent, run: mnemosyne setup --cron")
        return

    try:
        # Check if already registered
        result = subprocess.run(
            [hermes, "cron", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if "mnemosyne-sync" in result.stdout:
            logger.info("mnemosyne-sync cron job already registered")
            return
    except Exception as e:
        logger.warning(f"Could not check existing cron jobs: {e}")

    prompt = f"""Sync new OpenClaw and Hermes sessions into Mnemosyne memory.

Run the sync script with the current Python interpreter:
{python_path} {sync_script} --source all --verbose

Only index new sessions (incremental). Report a brief summary of how many sessions were indexed."""

    schedule = f"*/{interval_minutes} * * * *"

    try:
        subprocess.run(
            [
                hermes, "cron", "add",
                "--name", "mnemosyne-sync",
                "--schedule", schedule,
                "--deliver", "local",
                "--",
                prompt.strip(),
            ],
            capture_output=False,
            timeout=15,
        )
        logger.info(
            f"Registered mnemosyne-sync cron job (every {interval_minutes} min)"
        )
    except Exception as e:
        logger.warning(f"Could not register cron job: {e}")
        logger.info("To register manually after hermes-agent is installed:")
        logger.info(f"  hermes cron add --name mnemosyne-sync --schedule '*/{interval_minutes} * * * *'")


def run():
    logger.info("Mnemosyne post-install: starting setup...")

    python_path = _find_mnemosyne_python()
    sync_script = Path(__file__).parent / "sync.py"

    # Step 1: Initial full sync
    run_initial_sync(python_path)

    # Step 2: Register cron job
    register_cron_job(python_path, sync_script)

    logger.info("Mnemosyne setup complete!")


if __name__ == "__main__":
    run()
