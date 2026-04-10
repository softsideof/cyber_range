"""FastAPI entry point for the CyberRange environment server."""

import sys
import os

# Ensure the project root is on sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cyber_range.server.app import app  # noqa: E402, F401


def main() -> None:
    """Entry point for direct execution via uv run or openenv serve."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
