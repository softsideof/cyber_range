"""CyberRange client for connecting to the environment server."""

from openenv.core.mcp_client import MCPToolClient


class CyberRangeEnv(MCPToolClient):
    """Client for connecting to a running CyberRange environment.

    Inherits: list_tools(), call_tool(), reset(), step() from MCPToolClient.

    Supported task_ids:
        - script_kiddie (easy): SSH brute-force attack
        - phishing_campaign (medium): Multi-host phishing with false positives
        - apt_lateral_movement (hard): Full APT kill chain
        - ransomware_outbreak (hard): Time-critical ransomware containment
        - supply_chain_compromise (hard): Trojanized software update
        - insider_threat_apt (nightmare): Dual simultaneous threats

    Usage::

        from cyber_range import CyberRangeEnv

        with CyberRangeEnv(base_url="https://keshav-005-cyber-range.hf.space").sync() as env:
            obs = env.reset(task_id="script_kiddie", seed=42)
            obs = env.step({"tool_name": "observe_network", "arguments": {}})
    """
    pass
