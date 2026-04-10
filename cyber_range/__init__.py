"""CyberRange — OpenEnv environment for SOC analyst training.

An adaptive cybersecurity simulation with MITRE ATT&CK-aligned
attack scenarios, realistic forensics, and multi-objective grading.
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from .client import CyberRangeEnv

# Gymnasium wrapper (optional — requires gymnasium>=0.29.1)
try:
    from .gym_wrapper import CyberRangeGymEnv, make_env
except ImportError:
    CyberRangeGymEnv = None
    make_env = None

__all__ = [
    "CyberRangeEnv",
    "CyberRangeGymEnv",
    "make_env",
    "CallToolAction",
    "ListToolsAction",
]
