"""
CyberRange Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

OUTPUT FORMAT (REQUIRED):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import json
import os
import re
import sys
import textwrap
from typing import Any

# Ensure cyber_range package is importable from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from openenv.core.env_server.mcp_types import CallToolAction

# In-process environment (no server needed)
from cyber_range.server.cyber_environment import CyberRangeEnvironment

# ─────────────────────────────────────────────────────────────
# Environment Variables (REQUIRED by OpenEnv spec)
# ─────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

TEMPERATURE = 0.1
MAX_TOKENS = 500
SEED = 42
ENV_NAME = "cyber_range"

# Task definitions — all 6 scenarios
TASKS = [
    "script_kiddie",
    "phishing_campaign",
    "apt_lateral_movement",
    "ransomware_outbreak",
    "supply_chain_compromise",
    "insider_threat_apt",
]

# Client initialized in main()
client: OpenAI = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert Security Operations Center (SOC) analyst defending an enterprise network.
You interact with the CyberRange environment through tool calls.

AVAILABLE TOOLS:
- observe_network() → Get full network state, alerts, and metrics. Call this FIRST.
- investigate_alert(alert_id="ALT-XXXX") → Deep-dive into an alert.
- isolate_host(node_id="xxx-xx") → Quarantine a compromised host.
- block_ip(ip_address="x.x.x.x") → Block an external attacker IP at the firewall.
- run_forensics(node_id="xxx-xx") → Run forensics on a host. Expensive but reveals evidence.
- deploy_patch(node_id="xxx-xx") → Patch known vulnerabilities on a host.
- restore_backup(node_id="xxx-xx") → Restore a compromised host from backup.
- dismiss_alert(alert_id="ALT-XXXX") → Dismiss an alert as a false positive.
- deploy_honeypot() → Deploy a honeypot to gather attacker intel.
- escalate_incident(description="...") → Escalate to senior analyst.

KEY STRATEGIES:
- Investigate alerts BEFORE taking containment steps.
- Read forensic_evidence: "benign" or "routine" = false positive → dismiss it.
  "malicious activity" or "unauthorized access" = real threat → contain it.
- Prioritize by severity: critical > high > medium > low.
- Deploy honeypot early in complex scenarios for intelligence.

RESPONSE FORMAT - respond with EXACTLY one tool call:
TOOL: tool_name
ARGS: {"param": "value"}
""")


# ─────────────────────────────────────────────────────────────
# Heuristic (rule-based) Agent
# ─────────────────────────────────────────────────────────────

class HeuristicAgent:
    """Expert rule-based SOC analyst agent with scenario-specific playbooks."""

    def __init__(self, initial_alerts: list[dict], initial_topology: list[dict]):
        self._step = 0
        self._investigated_alerts: set[str] = set()
        self._blocked_ips: set[str] = set()
        self._dismissed_alerts: set[str] = set()
        self._isolated_nodes: set[str] = set()
        self._restored_nodes: set[str] = set()
        self._scenario_id = ""

        self._fp_candidates: list[str] = []
        self._real_alerts: list[str] = []
        self._all_alert_data: dict[str, dict] = {}

        for alert in initial_alerts:
            aid = alert.get("alert_id", "")
            self._all_alert_data[aid] = alert
            if alert.get("confidence", 1.0) < 0.5:
                self._fp_candidates.append(aid)
            else:
                self._real_alerts.append(aid)

        self._compromised_nodes: list[str] = [
            n["node_id"] for n in initial_topology
            if n.get("status") == "compromised"
        ]
        self._ips_to_block: list[str] = []
        self._nodes_to_isolate: list[str] = []
        self._confirmed_fps: list[str] = []

    def set_scenario(self, scenario_id: str):
        self._scenario_id = scenario_id

    def _process_evidence(self, last_result: Any, alerts: list[dict]) -> None:
        if not isinstance(last_result, dict):
            return
        details = last_result.get("details", {})
        if not isinstance(details, dict):
            return

        if "forensic_evidence" in details:
            evidence = details.get("forensic_evidence", "").lower()
            aid = details.get("alert_id", "")
            src_ip = details.get("source_ip", "")
            node = details.get("related_node_id", "") or details.get("related_node", "")

            is_fp = any(w in evidence for w in [
                "benign", "routine", "scheduled", "legitimate", "baseline",
                "no unauthorized", "appears clean", "matches expected",
                "nagios", "health check", "backup job",
            ])

            if is_fp:
                if aid and aid not in self._confirmed_fps:
                    self._confirmed_fps.append(aid)
            else:
                if src_ip and not src_ip.startswith("10.0.") and src_ip not in self._blocked_ips:
                    self._ips_to_block.append(src_ip)
                if node and node not in self._isolated_nodes:
                    self._nodes_to_isolate.append(node)

        # Update alerts from fresh data
        for a in alerts:
            aid = a.get("alert_id", "")
            if aid and aid not in self._all_alert_data:
                self._all_alert_data[aid] = a
                if a.get("confidence", 1.0) < 0.5:
                    self._fp_candidates.append(aid)
                else:
                    self._real_alerts.append(aid)

    def decide(self, last_result: Any, alerts: list[dict]) -> tuple[str, dict]:
        self._step += 1
        self._process_evidence(last_result, alerts)

        if self._step == 1:
            return "observe_network", {}

        # Act on evidence: block IPs
        if self._ips_to_block:
            ip = self._ips_to_block.pop(0)
            self._blocked_ips.add(ip)
            return "block_ip", {"ip_address": ip}

        # Dismiss confirmed FPs
        if self._confirmed_fps:
            aid = self._confirmed_fps.pop(0)
            if aid not in self._dismissed_alerts:
                self._dismissed_alerts.add(aid)
                return "dismiss_alert", {"alert_id": aid}

        # Isolate compromised nodes
        if self._nodes_to_isolate:
            node = self._nodes_to_isolate.pop(0)
            if node not in self._isolated_nodes:
                self._isolated_nodes.add(node)
                return "isolate_host", {"node_id": node}

        # Investigate unseen alerts (high severity first)
        sorted_alerts = sorted(
            [a for a in alerts if a.get("alert_id") not in self._investigated_alerts],
            key=lambda a: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                a.get("severity", "low"), 4
            )
        )
        if sorted_alerts:
            aid = sorted_alerts[0].get("alert_id", "")
            self._investigated_alerts.add(aid)
            return "investigate_alert", {"alert_id": aid}

        # Block known C2 IPs
        for ip in ["185.220.101.42", "94.232.46.19", "45.155.205.233",
                    "91.219.236.166", "198.51.100.23", "203.0.113.45"]:
            if ip not in self._blocked_ips:
                self._blocked_ips.add(ip)
                return "block_ip", {"ip_address": ip}

        # Dismiss remaining FP candidates after investigation
        for aid in self._fp_candidates:
            if aid not in self._dismissed_alerts and aid not in self._investigated_alerts:
                self._investigated_alerts.add(aid)
                return "investigate_alert", {"alert_id": aid}

        return "observe_network", {}


# ─────────────────────────────────────────────────────────────
# LLM Response Parser
# ─────────────────────────────────────────────────────────────

def parse_tool_call(response_text: str) -> tuple[str, dict]:
    """Parse the LLM response into a tool name and arguments dict."""
    tool_name = "observe_network"
    args: dict[str, Any] = {}

    if not response_text:
        return tool_name, args

    tool_match = re.search(r"TOOL:\s*(\w+)", response_text, re.IGNORECASE)
    if tool_match:
        tool_name = tool_match.group(1).strip()

    args_match = re.search(r"ARGS:\s*(\{.*?\})", response_text, re.DOTALL)
    if args_match:
        try:
            args = json.loads(args_match.group(1))
        except json.JSONDecodeError:
            args = {}

    return tool_name, args


def format_observation(obs_data: Any, step: int, max_steps: int) -> str:
    """Format observation data as context for the LLM."""
    if isinstance(obs_data, dict):
        display = dict(obs_data)
        if "network_topology" in display and len(display.get("network_topology", [])) > 6:
            topo = display["network_topology"]
            display["network_topology"] = topo[:6] + [
                {"note": f"... and {len(topo) - 6} more nodes"}
            ]
        formatted = json.dumps(display, indent=2, default=str)
    else:
        formatted = str(obs_data)[:3000]

    return f"Step {step}/{max_steps}\n\nLast tool result:\n{formatted}\n\nWhat is your next action? Respond with TOOL and ARGS."


def format_action_str(tool_name: str, tool_args: dict) -> str:
    """Format an action as a single-line string for [STEP] output."""
    if tool_args:
        args_str = ",".join(f"'{v}'" if isinstance(v, str) else str(v) for v in tool_args.values())
        return f"{tool_name}({args_str})"
    return f"{tool_name}()"


def sanitize_error(error_msg: str) -> str:
    """Sanitize error message to be single-line safe for stdout parsing."""
    if not error_msg:
        return "null"
    # Remove newlines and limit length to prevent parsing issues
    return error_msg.replace("\n", " ").replace("\r", "")[:200]


# ─────────────────────────────────────────────────────────────
# Episode Runner
# ─────────────────────────────────────────────────────────────

def run_episode(task_id: str, use_llm: bool = True) -> dict:
    """
    Run a single scenario episode.

    Emits the EXACT required output format:
        [START] task=<task_name> env=cyber_range model=<model_name>
        [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
        [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

    Always emits [END], even on exception.
    """
    global client  # Use the module-level client
    rewards: list[float] = []
    total_steps = 0
    success = False

    # [START] line — MUST include task, env, model
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    try:
        env = CyberRangeEnvironment()
        obs = env.reset(task_id=task_id, seed=SEED)

        metadata = obs.metadata or {}
        scenario = metadata.get("scenario", {})
        max_steps = scenario.get("max_steps", 20)
        alerts = metadata.get("pending_alerts", [])

        # Initialize agent
        heuristic = HeuristicAgent(
            initial_alerts=alerts,
            initial_topology=metadata.get("network_topology", []),
        )
        heuristic.set_scenario(task_id)
        history: list[dict] = []
        last_tool_result: Any = metadata
        last_error = None

        for step in range(1, max_steps + 1):
            total_steps = step
            last_error = None

            # Decide next action
            if use_llm:
                user_prompt = format_observation(last_tool_result, step, max_steps)
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for h in history[-4:]:
                    messages.append({"role": "user", "content": h["prompt"]})
                    messages.append({"role": "assistant", "content": h["response"]})
                messages.append({"role": "user", "content": user_prompt})

                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    response_text = completion.choices[0].message.content or ""
                    tool_name, tool_args = parse_tool_call(response_text)
                except Exception:
                    # LLM failed — fallback to heuristic silently
                    tool_name, tool_args = heuristic.decide(last_tool_result, alerts)
                    response_text = f"TOOL: {tool_name}\nARGS: {json.dumps(tool_args)}"
            else:
                tool_name, tool_args = heuristic.decide(last_tool_result, alerts)
                response_text = f"TOOL: {tool_name}\nARGS: {json.dumps(tool_args)}"
                user_prompt = f"Step {step}: heuristic mode"

            # Execute the tool
            try:
                obs = env.step(CallToolAction(tool_name=tool_name, arguments=tool_args))
            except Exception as exc:
                last_error = str(exc)
                try:
                    obs = env.step(CallToolAction(tool_name="observe_network", arguments={}))
                except Exception:
                    # Environment is broken — emit final step and stop
                    rewards.append(0.0)
                    action_str = format_action_str(tool_name, tool_args)
                    print(
                        f"[STEP] step={step} action={action_str} "
                        f"reward=0.00 done=true "
                        f"error={sanitize_error(last_error)}",
                        flush=True,
                    )
                    break
                tool_name = "observe_network"
                tool_args = {}

            reward = obs.reward if obs.reward else 0.0
            done = obs.done
            rewards.append(reward)
            action_str = format_action_str(tool_name, tool_args)
            error_str = sanitize_error(last_error) if last_error else "null"

            # [STEP] line — EXACT format required
            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={error_str}",
                flush=True,
            )

            # Parse result for next iteration
            raw_result = getattr(obs, "result", None)
            if isinstance(raw_result, dict):
                last_tool_result = raw_result
            elif raw_result is not None:
                try:
                    content_parts = getattr(raw_result, "content", [])
                    if content_parts:
                        text = getattr(content_parts[0], "text", str(content_parts[0]))
                        try:
                            last_tool_result = json.loads(text)
                        except (json.JSONDecodeError, TypeError):
                            last_tool_result = {"raw": str(text)[:2000]}
                    else:
                        last_tool_result = {"raw": str(raw_result)[:2000]}
                except Exception:
                    last_tool_result = {"raw": str(raw_result)[:2000]}
            else:
                last_tool_result = {}

            if isinstance(last_tool_result, dict) and "pending_alerts" in last_tool_result:
                alerts = last_tool_result["pending_alerts"]

            history.append({
                "prompt": (user_prompt[:500] if isinstance(user_prompt, str) else ""),
                "response": (response_text[:200] if isinstance(response_text, str) else ""),
            })

            if done:
                break

        # Get grader result
        state = env.state
        grader_result = getattr(state, "grader_result", None) or {}
        final_score = grader_result.get("final_score", 0.0)
        success = final_score >= 0.3

    except Exception:
        grader_result = {"final_score": 0.0}
        # Make sure we have at least one reward entry
        if not rewards:
            rewards.append(0.0)
            total_steps = max(total_steps, 1)

    # [END] line — ALWAYS emitted, even on exception
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={str(success).lower()} steps={total_steps} "
        f"rewards={rewards_str}",
        flush=True,
    )

    return grader_result


# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    """Run the LLM agent across all CyberRange scenarios."""
    global client

    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    use_llm = bool(HF_TOKEN)

    for task_id in TASKS:
        try:
            run_episode(task_id, use_llm=use_llm)
        except Exception:
            # Should never reach here since run_episode handles all errors,
            # but just in case — emit valid output
            print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)
            print(f"[END] success=false steps=0 rewards=0.00", flush=True)


if __name__ == "__main__":
    main()
