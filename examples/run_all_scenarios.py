"""
Example: Run the CyberRange agent on all scenarios.

Usage:
    # In-process (inside Docker container):
    python examples/run_all_scenarios.py

    # Remote (connecting to HF Space):
    ENV_BASE_URL=https://keshav-005-cyber-range.hf.space python examples/run_all_scenarios.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cyber_range.server.cyber_environment import CyberRangeEnvironment
from openenv.core.env_server.mcp_types import CallToolAction


SCENARIOS = [
    "script_kiddie",
    "phishing_campaign",
    "apt_lateral_movement",
    "ransomware_outbreak",
    "supply_chain_compromise",
    "insider_threat_apt",
]


def run_scenario(scenario_id: str, seed: int = 42) -> dict:
    """Run a single scenario using observe_network → investigate → contain."""
    env = CyberRangeEnvironment()
    obs = env.reset(task_id=scenario_id, seed=seed)

    metadata = obs.metadata or {}
    max_steps = metadata.get("scenario", {}).get("max_steps", 20)
    alerts = metadata.get("pending_alerts", [])

    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario_id} | Max Steps: {max_steps} | Alerts: {len(alerts)}")
    print(f"{'='*60}")

    # Step 1: Observe
    obs = env.step(CallToolAction(tool_name="observe_network", arguments={}))
    print(f"  Step 1: observe_network -> reward={obs.reward:.2f}")

    # Step 2+: Investigate all alerts
    for alert in alerts[:3]:
        aid = alert.get("alert_id", "")
        obs = env.step(CallToolAction(tool_name="investigate_alert", arguments={"alert_id": aid}))
        print(f"  Step {env.state.step_count}: investigate_alert({aid}) -> reward={obs.reward:.2f}")
        if obs.done:
            break

    # Final
    state = env.state
    grader = getattr(state, "grader_result", None) or {}
    score = grader.get("final_score", 0.01)
    print(f"  Result: score={score:.2f} | steps={state.step_count}")
    return grader


if __name__ == "__main__":
    print("CyberRange — Running All Scenarios")
    print("=" * 60)

    scores = {}
    for scenario_id in SCENARIOS:
        result = run_scenario(scenario_id)
        scores[scenario_id] = result.get("final_score", 0.01)

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for sid, score in scores.items():
        bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
        print(f"  {sid:30s} {bar} {score:.2f}")
    print(f"\n  Average: {sum(scores.values()) / len(scores):.2f}")

