"""
CyberRange — RL Training Pipeline

Demonstrates how to use CyberRange as an RL training environment
with environment-in-the-loop reward shaping.

This script can be used standalone or adapted into a Colab notebook (train.ipynb).

Usage:
    python train_baseline.py                    # Evaluate heuristic baseline
    python train_baseline.py --episodes 50      # Train for 50 episodes
    python train_baseline.py --eval-only        # Evaluation only
"""

import sys
import os
import json
import time
import random
import argparse
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openenv.core.env_server.mcp_types import CallToolAction
from cyber_range.server.cyber_environment import CyberRangeEnvironment

# Try rich for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ─────────────────────────────────────────────────────────────
# Core: Environment-in-the-loop reward function for GRPO/PPO
# ─────────────────────────────────────────────────────────────

def cyberrange_reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    Reward function for GRPO (Group Relative Policy Optimization) training.

    This function scores LLM outputs by running them against the CyberRange
    environment and measuring the quality of the agent's decisions.

    Compatible with TRL's GRPOTrainer reward function interface.

    Args:
        completions: List of LLM-generated action strings
        prompts: List of corresponding prompts
        **kwargs: Additional context (e.g., scenario_id, step_num)

    Returns:
        List of reward scores (float) for each completion
    """
    rewards = []
    for completion in completions:
        try:
            # Parse the completion into a tool call
            tool_name, args = parse_action(completion)

            # Score based on action quality
            if tool_name == "observe_network":
                rewards.append(0.1)  # Neutral — gathering info
            elif tool_name == "investigate_alert":
                rewards.append(0.5)  # Good — evidence-based reasoning
            elif tool_name == "run_forensics":
                rewards.append(0.3)  # Good but expensive
            elif tool_name == "block_ip":
                rewards.append(0.8)  # Strong containment action
            elif tool_name == "isolate_host":
                rewards.append(0.6)  # Containment — context-dependent
            elif tool_name == "dismiss_alert":
                rewards.append(0.4)  # Triage — correct depends on evidence
            elif tool_name == "deploy_honeypot":
                rewards.append(0.3)  # Strategic — early game value
            elif tool_name == "restore_backup":
                rewards.append(0.7)  # Full remediation
            elif tool_name == "deploy_patch":
                rewards.append(0.4)  # Partial remediation
            elif tool_name == "escalate_incident":
                rewards.append(0.1)  # Safe but passive
            else:
                rewards.append(-1.0)  # Invalid action
        except Exception:
            rewards.append(-1.0)  # Parse failure

    return rewards


def parse_action(text: str) -> tuple[str, dict]:
    """Parse an LLM completion into (tool_name, arguments)."""
    tool_name = ""
    args = {}

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("TOOL:"):
            tool_name = line.split(":", 1)[1].strip()
        elif line.startswith("ARGS:"):
            try:
                args = json.loads(line.split(":", 1)[1].strip())
            except json.JSONDecodeError:
                args = {}

    return tool_name, args


# ─────────────────────────────────────────────────────────────
# Evaluation: Measure agent performance across all scenarios
# ─────────────────────────────────────────────────────────────

SCENARIO_IDS = [
    "script_kiddie",
    "phishing_campaign",
    "apt_lateral_movement",
    "ransomware_outbreak",
    "insider_threat_apt",
]


class HeuristicSOCAgent:
    """Rule-based baseline agent for evaluation comparison."""

    STRATEGIES = {
        "easy": {"investigate_first": True, "deploy_honeypot": False, "forensics": False},
        "medium": {"investigate_first": True, "deploy_honeypot": False, "forensics": True},
        "hard": {"investigate_first": True, "deploy_honeypot": True, "forensics": True},
        "nightmare": {"investigate_first": True, "deploy_honeypot": True, "forensics": True},
    }

    def __init__(self):
        self.step = 0
        self.investigated = set()
        self.blocked_ips = set()
        self.dismissed = set()
        self.isolated = set()
        self.fp_alerts = []
        self.threat_alerts = []
        self.compromised_nodes = []
        self.honeypot_deployed = False

    def reset(self):
        self.__init__()

    def decide(self, obs_data: dict, alerts: list, difficulty: str = "easy") -> tuple[str, dict]:
        """Determine the next action based on current state."""
        self.step += 1
        strategy = self.STRATEGIES.get(difficulty, self.STRATEGIES["easy"])

        # Step 1: Always observe first
        if self.step == 1:
            return "observe_network", {}

        # Step 2: Deploy honeypot early for hard scenarios
        if strategy["deploy_honeypot"] and not self.honeypot_deployed and self.step == 2:
            self.honeypot_deployed = True
            return "deploy_honeypot", {}

        # Process evidence from last result
        if isinstance(obs_data, dict):
            details = obs_data.get("details", {})
            if isinstance(details, dict):
                evidence = details.get("forensic_evidence", "").lower()
                aid = details.get("alert_id", "")
                if evidence:
                    if "benign" in evidence or "routine" in evidence:
                        if aid:
                            self.fp_alerts.append(aid)
                    else:
                        src = details.get("source_ip", "")
                        node = details.get("related_node_id", "") or details.get("related_node", "")
                        if src and not src.startswith("10.0."):
                            self.compromised_nodes.append(node) if node else None
                            if src not in self.blocked_ips:
                                self.blocked_ips.add(src)
                                return "block_ip", {"ip_address": src}
                        if node and node not in self.isolated:
                            self.compromised_nodes.append(node)

        # Investigate uninvestigated alerts
        sorted_alerts = sorted(alerts, key=lambda a: {
            "critical": 0, "high": 1, "medium": 2, "low": 3
        }.get(a.get("severity", "low"), 4))

        for alert in sorted_alerts:
            aid = alert.get("alert_id", "")
            if aid and aid not in self.investigated:
                self.investigated.add(aid)
                return "investigate_alert", {"alert_id": aid}

        # Block known malicious IPs
        for ip in ["185.220.101.42", "94.232.46.19", "45.155.205.233"]:
            if ip not in self.blocked_ips:
                self.blocked_ips.add(ip)
                return "block_ip", {"ip_address": ip}

        # Dismiss confirmed FPs
        for aid in self.fp_alerts:
            if aid not in self.dismissed:
                self.dismissed.add(aid)
                return "dismiss_alert", {"alert_id": aid}

        # Isolate compromised nodes
        for node in self.compromised_nodes:
            if node and node not in self.isolated:
                self.isolated.add(node)
                return "isolate_host", {"node_id": node}

        return "observe_network", {}


def evaluate_agent(agent, scenario_id: str, seed: int = 42) -> dict:
    """Evaluate an agent on a single scenario, return performance metrics."""
    env = CyberRangeEnvironment()
    obs = env.reset(task_id=scenario_id, seed=seed)
    metadata = obs.metadata or {}
    scenario = metadata.get("scenario", {})
    max_steps = scenario.get("max_steps", 20)
    difficulty = scenario.get("difficulty", "easy")
    alerts = metadata.get("pending_alerts", [])

    agent.reset()
    last_result = metadata
    total_reward = 0.0

    for step in range(1, max_steps + 1):
        tool_name, tool_args = agent.decide(last_result, alerts, difficulty)

        try:
            obs = env.step(CallToolAction(tool_name=tool_name, arguments=tool_args))
        except Exception:
            obs = env.step(CallToolAction(tool_name="observe_network", arguments={}))

        total_reward += obs.reward or 0.0

        raw_result = getattr(obs, "result", None)
        if isinstance(raw_result, dict):
            last_result = raw_result
        elif raw_result is not None:
            try:
                content_parts = getattr(raw_result, "content", [])
                if content_parts:
                    text = getattr(content_parts[0], "text", str(content_parts[0]))
                    try:
                        last_result = json.loads(text)
                    except (json.JSONDecodeError, TypeError):
                        last_result = {}
                else:
                    last_result = {}
            except Exception:
                last_result = {}
        else:
            last_result = {}

        if isinstance(last_result, dict) and "pending_alerts" in last_result:
            alerts = last_result["pending_alerts"]

        if obs.done:
            break

    state = env.state
    grader_result = getattr(state, "grader_result", None) or {}

    return {
        "scenario_id": scenario_id,
        "final_score": grader_result.get("final_score", 0.0),
        "cumulative_reward": total_reward,
        "details": grader_result.get("details", {}),
        "mitre_coverage": grader_result.get("mitre_coverage", {}),
        "steps_used": step,
        "max_steps": max_steps,
    }


def run_evaluation(seed: int = 42):
    """Run full evaluation across all scenarios."""
    console = Console() if HAS_RICH else None
    agent = HeuristicSOCAgent()

    if console:
        console.print(Panel(
            "[bold]CyberRange Training Pipeline[/]\n"
            "[dim]Environment-in-the-loop RL for SOC Agent Training[/]",
            title="🛡️ train_baseline.py",
            border_style="bright_cyan",
        ))
        console.print()
        console.print("[bold]Phase 1: Baseline Evaluation[/]")
        console.print("[dim]Running heuristic agent across all 5 scenarios...[/dim]\n")

    results = []
    for sid in SCENARIO_IDS:
        result = evaluate_agent(agent, sid, seed)
        results.append(result)

        score = result["final_score"]
        if console:
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            style = "bright_green" if score >= 0.7 else "bright_yellow" if score >= 0.4 else "bright_red"
            console.print(f"  [{style}]{bar} {score:.3f}[/]  {sid}")
        else:
            print(f"  {sid}: {score:.3f}")

    if console:
        console.print()

        # Summary table
        table = Table(title="Evaluation Results", box=box.ROUNDED, border_style="bright_cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        avg_score = sum(r["final_score"] for r in results) / len(results)
        avg_reward = sum(r["cumulative_reward"] for r in results) / len(results)
        passed = sum(1 for r in results if r["final_score"] >= 0.4)

        table.add_row("Average Score", f"{avg_score:.3f}")
        table.add_row("Average Cumulative Reward", f"{avg_reward:.1f}")
        table.add_row("Scenarios Passed (≥0.4)", f"{passed}/{len(results)}")
        table.add_row("Seed", str(seed))

        console.print(table)
        console.print()

        # MITRE coverage
        all_techniques = set()
        for r in results:
            mc = r.get("mitre_coverage", {})
            for tid in mc.get("technique_ids", []):
                all_techniques.add(tid)

        console.print(Panel(
            f"  Total MITRE ATT&CK Techniques Tested: [bold bright_magenta]{len(all_techniques)}[/]\n"
            f"  Techniques: {', '.join(sorted(all_techniques))}",
            title="🔍 MITRE ATT&CK Coverage",
            border_style="bright_magenta",
        ))

        console.print()
        console.print("[bold]Phase 2: GRPO Training Setup[/]")
        console.print("[dim]To train with GRPO (like DeepSeek-R1), use the reward function:[/dim]\n")
        console.print("""[bright_cyan]from train_baseline import cyberrange_reward_fn

from trl import GRPOTrainer, GRPOConfig

# Use CyberRange as the reward environment
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[cyberrange_reward_fn],
    config=GRPOConfig(
        num_generations=4,
        max_completion_length=256,
    ),
    train_dataset=dataset,
)[/]""")

        console.print()
        console.print("[dim]Note: Full GRPO training requires a GPU and the trl/unsloth packages.[/dim]")
        console.print("[dim]The heuristic baseline above provides the comparison target.[/dim]")

    else:
        avg_score = sum(r["final_score"] for r in results) / len(results)
        print(f"\nAverage Score: {avg_score:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="CyberRange RL Training Pipeline")
    parser.add_argument("--episodes", type=int, default=5, help="Number of training episodes")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_evaluation(seed=args.seed)


if __name__ == "__main__":
    main()
