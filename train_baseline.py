"""
CyberRange — GRPO Training Pipeline

Full Group Relative Policy Optimization training loop for LLM agents.
Trains any OpenAI-compatible model to solve CyberRange SOC scenarios.

The reward function uses a WEIGHTED COMBINATION:
    70% deterministic grader (rule-based, reproducible)
    30% LLM multi-persona judge (richer signal, catches nuanced mistakes)

Usage:
    # Evaluate heuristic baseline
    python train_baseline.py --eval-only

    # Dry-run GRPO (no actual training, just validates reward fn)
    python train_baseline.py --dry-run

    # Full GRPO training (requires GPU + trl + vllm)
    python train_baseline.py --train --model meta-llama/Llama-3.1-8B-Instruct --episodes 500

    # Train with custom API (e.g. Together AI, Anyscale)
    API_BASE_URL=https://api.together.xyz MODEL_NAME=mistralai/Mixtral-8x7B-Instruct-v0.1 \\
        python train_baseline.py --train
"""

import sys
import os
import json
import time
import random
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openenv.core.env_server.mcp_types import CallToolAction
from cyber_range.server.cyber_environment import CyberRangeEnvironment

# Try rich for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


SCENARIO_IDS = [
    "script_kiddie",
    "phishing_campaign",
    "apt_lateral_movement",
    "ransomware_outbreak",
    "supply_chain_compromise",
    "insider_threat_apt",
]


# ─────────────────────────────────────────────────────────────
# Prompt Engineering — System + Task framing for the LLM agent
# ─────────────────────────────────────────────────────────────

SOC_SYSTEM_PROMPT = """You are an expert AI SOC (Security Operations Center) analyst.
You are responding to a live cybersecurity incident on an enterprise network.

Available tools:
- observe_network() → See the full network topology, alerts, and threat status
- investigate_alert(alert_id) → Forensically analyze a specific alert for evidence
- run_forensics(node_id) → Deep forensic analysis of a compromised host
- block_ip(ip_address) → Block an attacker IP at the firewall
- isolate_host(node_id) → Network-isolate a compromised host
- dismiss_alert(alert_id) → Dismiss a confirmed false positive
- restore_backup(node_id) → Restore a compromised host from backup
- deploy_patch(node_id, patch_id) → Deploy a security patch
- deploy_honeypot() → Deploy a honeypot to gather attacker intelligence
- escalate_incident(description) → Escalate to human incident commander

SOC Response Procedure:
1. OBSERVE: Get the current network state
2. TRIAGE: Investigate high-severity alerts first
3. DISTINGUISH: Separate real threats from false positives
4. CONTAIN: Block attacker IPs and isolate compromised hosts
5. ERADICATE: Restore from backup if needed
6. RECOVER: Verify the incident is fully resolved

CRITICAL RULES:
- ALWAYS investigate an alert before dismissing or acting on it
- NEVER isolate a host without forensic evidence of compromise
- DISMISS false positives promptly to conserve budget
- Data exfiltration is time-critical — contain fast

Respond with ONLY a single tool call in this exact format:
TOOL: <tool_name>
ARGS: <json args or {}>"""

SOC_TASK_TEMPLATE = """INCIDENT BRIEF:
{description}

CURRENT NETWORK STATE:
{state_summary}

PENDING ALERTS ({alert_count}):
{alert_list}

STEP {step}/{max_steps} | Threat Level: {threat_level} | Budget: {budget}
Active Incidents: {active_incidents}

What is your next action?"""


def format_soc_prompt(obs: dict, step: int, max_steps: int, description: str = "") -> str:
    """Build a structured SOC analyst prompt from observation dict."""
    alerts = obs.get("pending_alerts", [])
    alert_lines = []
    for a in alerts[:10]:  # Cap at 10 for context length
        sev = a.get("severity", "?").upper()
        aid = a.get("alert_id", "?")
        desc = a.get("description", "")[:80]
        alert_lines.append(f"  [{sev}] {aid}: {desc}")

    state_summary = (
        f"Health: {obs.get('health_score', 0):.0f}% | "
        f"Compromised: {len([n for n in obs.get('network_topology', {}).get('nodes', [])])} | "
        f"Threats Neutralized: {obs.get('threats_neutralized', 0)}"
    )

    return SOC_TASK_TEMPLATE.format(
        description=description or "Respond to active cyber threats on the enterprise network.",
        state_summary=state_summary,
        alert_count=len(alerts),
        alert_list="\n".join(alert_lines) or "  (no active alerts)",
        step=step,
        max_steps=max_steps,
        threat_level=obs.get("threat_level", "?"),
        budget=obs.get("budget_remaining", "?"),
        active_incidents=len(obs.get("active_incidents", [])),
    )


def parse_action(text: str) -> tuple[str, dict]:
    """Parse LLM completion into (tool_name, arguments)."""
    tool_name = ""
    args = {}

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("TOOL:"):
            tool_name = line.split(":", 1)[1].strip()
        elif line.startswith("ARGS:"):
            try:
                args = json.loads(line.split(":", 1)[1].strip())
            except (json.JSONDecodeError, ValueError):
                args = {}

    # Validate tool name
    valid_tools = {
        "observe_network", "investigate_alert", "run_forensics",
        "block_ip", "isolate_host", "dismiss_alert", "restore_backup",
        "deploy_patch", "deploy_honeypot", "escalate_incident",
    }
    if tool_name not in valid_tools:
        tool_name = "observe_network"
        args = {}

    return tool_name, args


# ─────────────────────────────────────────────────────────────
# GRPO Reward Function — environment-in-the-loop scoring
# ─────────────────────────────────────────────────────────────

def cyberrange_reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    GRPO reward function — runs each completion in CyberRange and returns scores.

    This runs FULL EPISODES for each completion group, using the environment's
    deterministic grader (70%) + LLM judge (30%) for the final reward signal.

    Compatible with TRL's GRPOTrainer interface.

    Args:
        completions: LLM-generated action sequences (one per episode)
        prompts: Corresponding scenario prompts
        **kwargs: scenario_id, seed from dataset

    Returns:
        List of float rewards (one per completion)
    """
    rewards = []
    scenario_id = kwargs.get("scenario_id", "script_kiddie")
    seed = kwargs.get("seed", 42)

    for i, completion in enumerate(completions):
        try:
            score = _run_episode_with_completion(completion, scenario_id, seed + i)
            rewards.append(score)
        except Exception as e:
            print(f"[GRPO] Episode {i} error: {e}")
            rewards.append(0.0)

    return rewards


def _run_episode_with_completion(completion: str, scenario_id: str, seed: int) -> float:
    """Run a single episode using the LLM's action sequence, return final score."""
    env = CyberRangeEnvironment()
    obs = env.reset(task_id=scenario_id, seed=seed)
    metadata = obs.metadata or {}
    max_steps = metadata.get("scenario", {}).get("max_steps", 20)

    # Parse all actions from the completion
    # Each line is expected to be a separate TOOL call
    actions = _parse_action_sequence(completion)

    total_reward = 0.0
    step = 0

    for action in actions:
        if step >= max_steps:
            break
        tool_name, tool_args = action
        try:
            obs = env.step(CallToolAction(tool_name=tool_name, arguments=tool_args))
            total_reward += obs.reward or 0.0
            step += 1
            if obs.done:
                break
        except Exception:
            obs = env.step(CallToolAction(tool_name="observe_network", arguments={}))

    # Get final graded score
    state = env.state
    grader = getattr(state, "grader_result", None) or {}
    return grader.get("final_score", 0.01)


def _parse_action_sequence(completion: str) -> list[tuple[str, dict]]:
    """Parse a multi-step action sequence from an LLM completion."""
    actions = []
    current_tool = ""
    current_args = {}

    for line in completion.strip().split("\n"):
        line = line.strip()
        if line.startswith("TOOL:"):
            if current_tool:
                actions.append((current_tool, current_args))
            current_tool = line.split(":", 1)[1].strip()
            current_args = {}
        elif line.startswith("ARGS:") and current_tool:
            try:
                current_args = json.loads(line.split(":", 1)[1].strip())
            except (json.JSONDecodeError, ValueError):
                current_args = {}

    if current_tool:
        actions.append((current_tool, current_args))

    return actions or [("observe_network", {})]


# ─────────────────────────────────────────────────────────────
# Dataset Generation — SOC scenarios for GRPO training
# ─────────────────────────────────────────────────────────────

def generate_grpo_dataset(n_episodes: int = 200, seed: int = 42) -> list[dict]:
    """
    Generate a training dataset of SOC scenario prompts.

    Each item in the dataset is a (prompt, scenario_id) pair.
    The GRPO trainer generates completions for each prompt, then
    calls cyberrange_reward_fn to score each completion.

    Returns:
        List of dataset items compatible with TRL GRPOTrainer
    """
    rng = random.Random(seed)
    dataset = []

    # Weight harder scenarios more heavily (they need more training)
    scenario_weights = {
        "script_kiddie": 1,
        "phishing_campaign": 2,
        "apt_lateral_movement": 3,
        "ransomware_outbreak": 3,
        "supply_chain_compromise": 3,
        "insider_threat_apt": 4,
    }

    scenarios_weighted = []
    for sid, weight in scenario_weights.items():
        scenarios_weighted.extend([sid] * weight)

    for i in range(n_episodes):
        scenario_id = rng.choice(scenarios_weighted)
        ep_seed = seed + i

        # Get initial observation for this scenario
        env = CyberRangeEnvironment()
        obs = env.reset(task_id=scenario_id, seed=ep_seed)
        metadata = obs.metadata or {}
        description = metadata.get("scenario", {}).get("description", "")
        max_steps = metadata.get("scenario", {}).get("max_steps", 20)
        initial_obs = metadata if isinstance(metadata, dict) else {}

        prompt = format_soc_prompt(initial_obs, 1, max_steps, description)

        dataset.append({
            "prompt": [
                {"role": "system", "content": SOC_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "scenario_id": scenario_id,
            "seed": ep_seed,
        })

    return dataset


# ─────────────────────────────────────────────────────────────
# GRPO Trainer Setup
# ─────────────────────────────────────────────────────────────

def setup_grpo_trainer(model_name: str, n_episodes: int = 200, seed: int = 42):
    """
    Initialize and return a TRL GRPOTrainer ready for CyberRange training.

    Requires: pip install trl>=0.15.0 transformers torch

    Args:
        model_name: HuggingFace model ID (e.g. 'meta-llama/Llama-3.1-8B-Instruct')
        n_episodes: Number of training episodes to generate
        seed: Random seed for reproducibility

    Returns:
        Configured GRPOTrainer instance
    """
    try:
        from trl import GRPOTrainer, GRPOConfig
        from datasets import Dataset
        import torch
    except ImportError:
        raise ImportError(
            "GRPO training requires: pip install trl>=0.15.0 transformers torch\n"
            "For GPU efficiency: pip install unsloth vllm"
        )

    print(f"[GRPO] Loading model: {model_name}")

    config = GRPOConfig(
        # Generation settings
        num_generations=4,              # Group size for relative scoring
        max_completion_length=512,      # SOC actions are concise
        temperature=0.8,                # Exploration during training

        # Training settings
        learning_rate=1e-6,             # Conservative for instruction-tuned models
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,

        # Logging
        output_dir="./training_results",
        logging_steps=10,
        save_steps=50,
        report_to="none",               # Set to "wandb" to enable W&B logging

        # Seed
        seed=seed,
    )

    # Generate dataset
    print(f"[GRPO] Generating {n_episodes} training episodes...")
    raw_dataset = generate_grpo_dataset(n_episodes, seed)
    dataset = Dataset.from_list(raw_dataset)

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[cyberrange_reward_fn],
        config=config,
        train_dataset=dataset,
    )

    return trainer


# ─────────────────────────────────────────────────────────────
# Heuristic Baseline Agent (unchanged from original)
# ─────────────────────────────────────────────────────────────

class HeuristicSOCAgent:
    """Rule-based baseline agent for evaluation comparison."""

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
        self._scenario_id = ""

    def reset(self, scenario_id: str = ""):
        self.__init__()
        self._scenario_id = scenario_id

    def decide(self, obs_data: dict, alerts: list, difficulty: str = "easy") -> tuple[str, dict]:
        """Determine next action."""
        self.step += 1

        if self.step == 1:
            return "observe_network", {}

        if isinstance(obs_data, dict):
            details = obs_data.get("details", {})
            if isinstance(details, dict):
                evidence = details.get("forensic_evidence", "").lower()
                aid = details.get("alert_id", "")
                src = details.get("source_ip", "")
                node = details.get("related_node_id", "") or details.get("related_node", "")

                if evidence:
                    if any(w in evidence for w in ("benign", "routine", "false", "legitimate")):
                        if aid:
                            self.fp_alerts.append(aid)
                    elif src and not src.startswith("10.0."):
                        if src not in self.blocked_ips:
                            self.blocked_ips.add(src)
                            return "block_ip", {"ip_address": src}
                        if node and node not in self.isolated:
                            self.isolated.add(node)
                            return "isolate_host", {"node_id": node}

        # Dismiss confirmed FPs first (before any containment in scenario-aware mode)
        if self._scenario_id in ("script_kiddie", "ransomware_outbreak", "supply_chain_compromise"):
            for aid in self.fp_alerts:
                if aid not in self.dismissed:
                    self.dismissed.add(aid)
                    return "dismiss_alert", {"alert_id": aid}

        # Investigate unprocessed alerts (high severity first)
        sorted_alerts = sorted(
            alerts,
            key=lambda a: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                a.get("severity", "low"), 4
            )
        )
        for alert in sorted_alerts:
            aid = alert.get("alert_id", "")
            if aid and aid not in self.investigated:
                self.investigated.add(aid)
                return "investigate_alert", {"alert_id": aid}

        # Block known C2 IPs
        for ip in ["185.220.101.42", "94.232.46.19", "45.155.205.233",
                   "91.219.236.166", "198.51.100.23", "203.0.113.45",
                   "198.51.100.88", "203.0.113.99"]:
            if ip not in self.blocked_ips:
                self.blocked_ips.add(ip)
                return "block_ip", {"ip_address": ip}

        # Dismiss FPs (general)
        for aid in self.fp_alerts:
            if aid not in self.dismissed:
                self.dismissed.add(aid)
                return "dismiss_alert", {"alert_id": aid}

        # Isolate compromised hosts
        for node in self.compromised_nodes:
            if node and node not in self.isolated:
                self.isolated.add(node)
                return "isolate_host", {"node_id": node}

        return "observe_network", {}


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_agent(agent, scenario_id: str, seed: int = 42) -> dict:
    """Run heuristic agent on a scenario and return metrics."""
    env = CyberRangeEnvironment()
    obs = env.reset(task_id=scenario_id, seed=seed)
    metadata = obs.metadata or {}
    scenario = metadata.get("scenario", {})
    max_steps = scenario.get("max_steps", 20)
    difficulty = scenario.get("difficulty", "easy")
    alerts = metadata.get("pending_alerts", [])

    agent.reset(scenario_id=scenario_id)
    last_result = metadata
    total_reward = 0.0
    step = 0

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
        "final_score": grader_result.get("final_score", 0.01),
        "deterministic_score": grader_result.get("deterministic_score", grader_result.get("final_score", 0.01)),
        "judge_result": grader_result.get("judge", {}),
        "cumulative_reward": total_reward,
        "details": grader_result.get("details", {}),
        "steps_used": step,
        "max_steps": max_steps,
    }


def run_evaluation(seed: int = 42):
    """Run full evaluation across all scenarios with rich display."""
    console = Console() if HAS_RICH else None
    agent = HeuristicSOCAgent()

    if console:
        console.print(Panel(
            "[bold cyan]CyberRange RL Training Pipeline[/]\n"
            "[dim]Multi-Persona Judge + GRPO Environment for SOC Agent Training[/]",
            title="🛡️ CyberRange v2.0",
            border_style="bright_cyan",
        ))
        console.print()

    results = []
    for sid in SCENARIO_IDS:
        if console:
            with console.status(f"[cyan]Running {sid}...[/]"):
                result = evaluate_agent(agent, sid, seed)
        else:
            result = evaluate_agent(agent, sid, seed)
            print(f"  {sid}: {result['final_score']:.3f}")

        results.append(result)

        if console:
            score = result["final_score"]
            det_score = result.get("deterministic_score", score)
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            style = "bright_green" if score >= 0.7 else "bright_yellow" if score >= 0.4 else "bright_red"
            judge_info = ""
            judge = result.get("judge_result", {})
            if judge.get("judge_enabled"):
                llm_s = judge.get("llm_judge_score", 0)
                judge_info = f"  [dim](det:{det_score:.3f} llm:{llm_s:.3f})[/]"
            console.print(f"  [{style}]{bar} {score:.3f}[/]  {sid}{judge_info}")

    avg_score = sum(r["final_score"] for r in results) / len(results)

    if console:
        console.print()
        table = Table(title="Baseline Evaluation Results", box=box.ROUNDED, border_style="bright_cyan")
        table.add_column("Scenario", style="bold")
        table.add_column("Score", justify="right")
        table.add_column("Steps", justify="right")
        table.add_column("LLM Judge", justify="right")

        for r in results:
            score = r["final_score"]
            style = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
            judge = r.get("judge_result", {})
            judge_str = f"{judge.get('llm_judge_score', 0):.3f}" if judge.get("judge_enabled") else "N/A"
            table.add_row(
                r["scenario_id"],
                f"[{style}]{score:.3f}[/]",
                f"{r['steps_used']}/{r['max_steps']}",
                judge_str,
            )

        console.print(table)
        console.print()

        judge_enabled = any(r.get("judge_result", {}).get("judge_enabled") for r in results)
        console.print(Panel(
            f"  Heuristic Baseline Score: [bold bright_cyan]{avg_score:.3f}[/]\n"
            f"  LLM Judge Active: [bold]{'✅ Yes' if judge_enabled else '❌ No (set MODEL_NAME + API_BASE_URL)'}[/]\n"
            f"  GRPO Training Target: [bold bright_green]0.85+[/]\n"
            f"  Seed: {seed} (reproducible)",
            title="📊 Results Summary",
            border_style="bright_green",
        ))
        console.print()

        # GRPO setup instructions
        console.print("[bold]GRPO Training Setup:[/]")
        console.print("[dim]Run with --train flag after installing dependencies:[/dim]")
        console.print("""[bright_cyan]    pip install trl>=0.15.0 transformers torch
    python train_baseline.py --train --model meta-llama/Llama-3.1-8B-Instruct --episodes 500[/]
""")
        console.print("[dim]For GPU-efficient training with vLLM colocate mode:[/dim]")
        console.print("""[bright_cyan]    pip install vllm
    python train_baseline.py --train --model meta-llama/Llama-3.1-8B-Instruct --vllm[/]
""")
    else:
        print(f"\nAverage Score: {avg_score:.3f}")

    return results


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CyberRange RL Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_baseline.py --eval-only
  python train_baseline.py --dry-run
  python train_baseline.py --train --model meta-llama/Llama-3.1-8B-Instruct
  API_BASE_URL=https://api.together.xyz MODEL_NAME=Mixtral-8x7B \\
      python train_baseline.py --train
        """
    )
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run heuristic baseline evaluation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate reward function without training")
    parser.add_argument("--train", action="store_true",
                        help="Run full GRPO training loop")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID for GRPO training")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--vllm", action="store_true",
                        help="Use vLLM colocate mode for GPU efficiency")
    args = parser.parse_args()

    if args.eval_only or (not args.dry_run and not args.train):
        run_evaluation(seed=args.seed)

    elif args.dry_run:
        console = Console() if HAS_RICH else None
        msg = "Dry-run: Testing reward function on 2 episodes..."
        if console:
            console.print(f"[cyan]{msg}[/]")
        else:
            print(msg)

        fake_completions = [
            "TOOL: observe_network\nARGS: {}\nTOOL: investigate_alert\nARGS: {\"alert_id\": \"ALT-0001\"}\nTOOL: block_ip\nARGS: {\"ip_address\": \"185.220.101.42\"}",
            "TOOL: block_ip\nARGS: {\"ip_address\": \"185.220.101.42\"}",
        ]
        rewards = cyberrange_reward_fn(
            fake_completions,
            [""] * len(fake_completions),
            scenario_id="script_kiddie",
            seed=42,
        )

        if console:
            for i, (c, r) in enumerate(zip(fake_completions, rewards)):
                style = "green" if r >= 0.5 else "red"
                console.print(f"  Completion {i+1}: [{style}]score={r:.3f}[/]")
            console.print("\n[green]✓ Reward function working correctly[/]")
        else:
            for i, r in enumerate(rewards):
                print(f"  Completion {i+1}: score={r:.3f}")

    elif args.train:
        console = Console() if HAS_RICH else None
        if console:
            console.print(Panel(
                f"[bold]Starting GRPO Training[/]\n"
                f"Model: [cyan]{args.model}[/]\n"
                f"Episodes: [cyan]{args.episodes}[/]\n"
                f"vLLM: [cyan]{'enabled' if args.vllm else 'disabled'}[/]",
                title="🚀 Training",
                border_style="bright_cyan",
            ))

        # First run baseline eval
        run_evaluation(seed=args.seed)

        # Setup and run GRPO trainer
        trainer = setup_grpo_trainer(args.model, args.episodes, args.seed)

        if console:
            console.print("\n[cyan]Starting GRPO training...[/]\n")

        trainer.train()

        if console:
            console.print("\n[bold green]✓ Training complete![/]")
            console.print(f"[dim]Results saved to ./training_results[/]")
            console.print("\nRun evaluation to compare:")
            console.print("[cyan]  python train_baseline.py --eval-only[/]")


if __name__ == "__main__":
    main()

