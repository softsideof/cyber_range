"""FastAPI entry point for the CyberRange environment server."""

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    from .cyber_environment import CyberRangeEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.cyber_environment import CyberRangeEnvironment

# Create the app with web interface and README integration
# Pass the class (factory) for WebSocket per-session environment instances
# Use MCP types since this is an MCP-based environment
app = create_app(
    CyberRangeEnvironment, CallToolAction, CallToolObservation,
    env_name="cyber_range",
)


def main() -> None:
    """
    Entry point for direct execution.

    Enables:
        uv run --project . server
        python -m cyber_range.server.app
        openenv serve cyber_range
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
