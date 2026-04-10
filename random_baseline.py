"""Run a random agent baseline to get real 'random' scores for the README."""
import sys, os, random, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cyber_range.server.cyber_environment import CyberRangeEnvironment
from openenv.core.env_server.mcp_types import CallToolAction

TASKS = ["script_kiddie", "phishing_campaign", "apt_lateral_movement", "ransomware_outbreak", "insider_threat_apt"]
TOOLS = ["observe_network", "investigate_alert", "isolate_host", "block_ip",
         "deploy_honeypot", "escalate_incident", "dismiss_alert"]

rng = random.Random(42)

print("RANDOM AGENT BASELINE (seed=42)")
print("=" * 50)

for task_id in TASKS:
    env = CyberRangeEnvironment()
    obs = env.reset(task_id=task_id, seed=42)
    meta = obs.metadata or {}
    max_steps = meta.get("scenario", {}).get("max_steps", 20)
    alerts = meta.get("pending_alerts", [])
    topo = meta.get("network_topology", [])
    alert_ids = [a["alert_id"] for a in alerts]
    node_ids = [n["node_id"] for n in topo]

    for step in range(max_steps):
        tool = rng.choice(TOOLS)
        args = {}
        if tool == "investigate_alert" and alert_ids:
            args = {"alert_id": rng.choice(alert_ids)}
        elif tool == "isolate_host" and node_ids:
            args = {"node_id": rng.choice(node_ids)}
        elif tool == "block_ip":
            args = {"ip_address": f"{rng.randint(1,255)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,255)}"}
        elif tool == "dismiss_alert" and alert_ids:
            args = {"alert_id": rng.choice(alert_ids)}
        elif tool == "escalate_incident":
            args = {"description": "random escalation"}

        try:
            r = env.step(CallToolAction(tool_name=tool, arguments=args))
        except Exception:
            r = env.step(CallToolAction(tool_name="observe_network", arguments={}))
        if r.done:
            break

    grader = getattr(env.state, "grader_result", {})
    score = grader.get("final_score", 0.0)
    print(f"  {task_id:<25} score={score:.3f}")

print()
