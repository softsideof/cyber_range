"""
CyberRange — Gymnasium-Compatible Wrapper

Wraps CyberRangeEnvironment as a proper Gymnasium environment with
Text observation and action spaces, enabling integration with standard
RL training frameworks (TRL, RLlib, Stable Baselines 3).

Usage:
    from cyber_range.gym_wrapper import make_env

    env = make_env(task_id="apt_lateral_movement", seed=42)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step('{"tool": "observe_network"}')
"""

import json
from typing import Any, Optional

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None  # type: ignore
    spaces = None  # type: ignore

from cyber_range.server.cyber_environment import CyberRangeEnvironment
from openenv.core.env_server.mcp_types import CallToolAction


VALID_TOOLS = [
    "observe_network", "investigate_alert", "isolate_host", "block_ip",
    "run_forensics", "deploy_patch", "restore_backup", "dismiss_alert",
    "deploy_honeypot", "escalate_incident",
]

# Base class — gym.Env if available, otherwise object
_BaseEnv = gym.Env if HAS_GYMNASIUM else object


class CyberRangeGymEnv(_BaseEnv):
    """
    Gymnasium wrapper for CyberRange.

    Observation space: Text (natural language SOC state)
    Action space: Text (JSON tool call)

    Compatible with:
    - TRL (Transformer Reinforcement Learning)
    - RLlib / OpenRL
    - Stable Baselines 3 (with Text wrapper)
    - GRPO / PPO policy optimization
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        task_id: str = "script_kiddie",
        seed: int = 42,
        max_obs_length: int = 4096,
        max_action_length: int = 512,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.task_id = task_id
        self.seed_val = seed
        self.render_mode = render_mode

        if HAS_GYMNASIUM and spaces is not None:
            self.observation_space = spaces.Text(
                min_length=0,
                max_length=max_obs_length,
            )
            self.action_space = spaces.Text(
                min_length=1,
                max_length=max_action_length,
            )

        self._env = CyberRangeEnvironment()
        self._step_count = 0
        self._max_steps = 20
        self._last_obs_text = ""

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[str, dict]:
        """Reset the environment and return initial observation."""
        actual_seed = seed if seed is not None else self.seed_val
        task = self.task_id
        if options and "task_id" in options:
            task = options["task_id"]

        obs = self._env.reset(task_id=task, seed=actual_seed)
        self._step_count = 0
        metadata = obs.metadata or {}
        scenario = metadata.get("scenario", {})
        self._max_steps = scenario.get("max_steps", 20)

        obs_text = self._format_observation(metadata)
        self._last_obs_text = obs_text

        info = {
            "scenario": scenario.get("name", task),
            "difficulty": scenario.get("difficulty", "easy"),
            "max_steps": self._max_steps,
            "tools_available": VALID_TOOLS,
        }
        return obs_text, info

    def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
        """
        Take an action and return (observation, reward, terminated, truncated, info).

        Action format: JSON string with "tool" and optional "args" keys.
        Example: '{"tool": "investigate_alert", "args": {"alert_id": "ALT-0001"}}'
        """
        self._step_count += 1

        # Parse action
        try:
            action_data = json.loads(action)
            tool_name = action_data.get("tool", "observe_network")
            tool_args = action_data.get("args", {})
        except (json.JSONDecodeError, AttributeError):
            # Try simple tool name
            tool_name = action.strip() if action.strip() in VALID_TOOLS else "observe_network"
            tool_args = {}

        if tool_name not in VALID_TOOLS:
            tool_name = "observe_network"

        # Execute
        try:
            obs = self._env.step(CallToolAction(tool_name=tool_name, arguments=tool_args))
        except Exception:
            obs = self._env.step(CallToolAction(tool_name="observe_network", arguments={}))

        reward = float(obs.reward or 0.0)
        terminated = bool(obs.done)
        truncated = self._step_count >= self._max_steps and not terminated

        # Format observation
        raw_result = getattr(obs, "result", None)
        result_data = self._extract_result(raw_result)
        obs_text = self._format_step_result(tool_name, result_data, self._step_count)
        self._last_obs_text = obs_text

        info = {
            "step": self._step_count,
            "reward": reward,
            "tool_used": tool_name,
            "raw_result": result_data,
        }

        if terminated or truncated:
            state = self._env.state
            grader = getattr(state, "grader_result", None) or {}
            info["final_score"] = grader.get("final_score", 0.0)
            info["grader_result"] = grader
            info["mitre_coverage"] = grader.get("mitre_coverage", {})

        return obs_text, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render the current state."""
        if self.render_mode == "ansi":
            return self._last_obs_text
        elif self.render_mode == "human":
            print(self._last_obs_text)
            return None
        return None

    def _extract_result(self, raw_result: Any) -> dict:
        """Extract dict from observation result."""
        if isinstance(raw_result, dict):
            return raw_result
        if raw_result is not None:
            try:
                content_parts = getattr(raw_result, "content", [])
                if content_parts:
                    text = getattr(content_parts[0], "text", str(content_parts[0]))
                    return json.loads(text)
            except Exception:
                pass
        return {}

    def _format_observation(self, metadata: dict) -> str:
        """Format initial observation as natural language."""
        scenario = metadata.get("scenario", {})
        alerts = metadata.get("pending_alerts", [])

        lines = [
            f"=== CyberRange: {scenario.get('name', 'Unknown Scenario')} ===",
            f"Difficulty: {scenario.get('difficulty', 'unknown')}",
            f"Max Steps: {scenario.get('max_steps', 20)}",
            f"",
            f"PENDING ALERTS ({len(alerts)}):",
        ]

        for alert in alerts[:5]:
            lines.append(
                f"  [{alert.get('severity', '?').upper()}] {alert.get('alert_id', '?')}: "
                f"{alert.get('description', '')[:100]}"
            )

        lines.extend([
            "",
            "Available tools: " + ", ".join(VALID_TOOLS),
            "",
            "Format your action as JSON: {\"tool\": \"tool_name\", \"args\": {\"param\": \"value\"}}",
        ])

        return "\n".join(lines)

    def _format_step_result(self, tool: str, result: dict, step: int) -> str:
        """Format step result as natural language."""
        lines = [f"[Step {step}/{self._max_steps}] Tool: {tool}"]

        if "forensic_evidence" in result:
            lines.append(f"Evidence: {str(result['forensic_evidence'])[:500]}")
        elif "process_tree" in result:
            lines.append(f"Forensic scan of {result.get('hostname', '?')}:")
            lines.append(f"  Malware found: {result.get('malware_found', False)}")
            lines.append(f"  Risk score: {result.get('risk_score', 0)}/100")
            procs = result.get("process_tree", [])
            suspicious = [p for p in procs if p.get("suspicious")]
            if suspicious:
                lines.append(f"  Suspicious processes: {len(suspicious)}")
        elif "network_summary" in result:
            summary = result["network_summary"]
            lines.append(f"  Threat Level: {summary.get('overall_threat_level', '?')}")
        else:
            desc = result.get("description", str(result)[:200])
            lines.append(f"  Result: {desc}")

        return "\n".join(lines)


def make_env(
    task_id: str = "script_kiddie",
    seed: int = 42,
    render_mode: Optional[str] = None,
) -> CyberRangeGymEnv:
    """Create a CyberRange Gymnasium environment."""
    if not HAS_GYMNASIUM:
        raise ImportError(
            "gymnasium is required for the Gym wrapper. "
            "Install with: pip install gymnasium>=0.29.1"
        )
    return CyberRangeGymEnv(task_id=task_id, seed=seed, render_mode=render_mode)
