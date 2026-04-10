"""CyberRange client for connecting to the environment server."""

from openenv.core.mcp_client import MCPToolClient


class CyberRangeEnv(MCPToolClient):
    """Client wrapper. Inherits list_tools(), call_tool(), reset(), step() from MCPToolClient.

    Supported task_ids: script_kiddie, phishing_campaign, apt_lateral_movement,
    ransomware_outbreak, insider_threat_apt.
    """
    pass
