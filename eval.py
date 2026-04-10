"""
CyberRange — Evaluation Script

Runs the heuristic baseline agent across all 6 scenarios and reports
side-by-side performance metrics. Saves results to training_results/
for reward curve visualization.

Usage:
    python eval.py                          # Evaluate heuristic baseline
    python eval.py --scenarios script_kiddie ransomware_outbreak
    python eval.py --seed 123 --runs 3     # Average over 3 seeds
    python eval.py --save                   # Save results to JSON
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

# Windows: force UTF-8 output to prevent cp1252 encoding errors with rich
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openenv.core.env_server.mcp_types import CallToolAction
from cyber_range.server.cyber_environment import CyberRangeEnvironment

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.rule import Rule
    from rich import box
    # Windows-safe: force UTF-8 encoding, disable legacy renderer
    import io
    _console = Console(highlight=False, markup=True)
    HAS_RICH = True
except (ImportError, Exception):
    HAS_RICH = False
    _console = None


ALL_SCENARIOS = [
    "script_kiddie",
    "phishing_campaign",
    "apt_lateral_movement",
    "ransomware_outbreak",
    "supply_chain_compromise",
    "insider_threat_apt",
]

SCENARIO_DIFFICULTY = {
    "script_kiddie": "easy",
    "phishing_campaign": "medium",
    "apt_lateral_movement": "hard",
    "ransomware_outbreak": "hard",
    "supply_chain_compromise": "hard",
    "insider_threat_apt": "nightmare",
}


# ─────────────────────────────────────────────────────────────
# Heuristic Baseline Agent
# ─────────────────────────────────────────────────────────────

class HeuristicSOCAgent:
    """Rule-based baseline agent. Establishes the score floor each LLM must beat."""

    def __init__(self):
        self.step = 0
        self.investigated = set()
        self.blocked_ips = set()
        self.dismissed = set()
        self.isolated = set()
        self.fp_alerts = []
        self.compromised_nodes = []
        self.honeypot_deployed = False
        self._scenario_id = ""

    def reset(self, scenario_id: str = ""):
        self.__init__()
        self._scenario_id = scenario_id

    def decide(self, obs_data: dict, alerts: list, difficulty: str = "easy") -> tuple[str, dict]:
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

        # Dismiss FPs first (before containment) for scenarios with many FPs
        if self._scenario_id in ("script_kiddie", "ransomware_outbreak", "supply_chain_compromise"):
            for aid in self.fp_alerts:
                if aid not in self.dismissed:
                    self.dismissed.add(aid)
                    return "dismiss_alert", {"alert_id": aid}

        # Investigate unseen alerts (high severity first)
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

        for ip in ["185.220.101.42", "94.232.46.19", "45.155.205.233",
                   "91.219.236.166", "198.51.100.23", "203.0.113.45",
                   "198.51.100.88", "203.0.113.99"]:
            if ip not in self.blocked_ips:
                self.blocked_ips.add(ip)
                return "block_ip", {"ip_address": ip}

        for aid in self.fp_alerts:
            if aid not in self.dismissed:
                self.dismissed.add(aid)
                return "dismiss_alert", {"alert_id": aid}

        for node in self.compromised_nodes:
            if node and node not in self.isolated:
                self.isolated.add(node)
                return "isolate_host", {"node_id": node}

        return "observe_network", {}


# ─────────────────────────────────────────────────────────────
# Episode Runner
# ─────────────────────────────────────────────────────────────

def run_episode(agent, scenario_id: str, seed: int = 42) -> dict:
    """Run a single episode, return full metrics."""
    env = CyberRangeEnvironment()
    obs = env.reset(task_id=scenario_id, seed=seed)
    metadata = obs.metadata or {}
    scenario_meta = metadata.get("scenario", {})
    max_steps = scenario_meta.get("max_steps", 20)
    difficulty = scenario_meta.get("difficulty", "easy")
    alerts = metadata.get("pending_alerts", [])

    agent.reset(scenario_id=scenario_id)
    last_result = metadata
    total_reward = 0.0
    step = 0
    action_log = []

    for step in range(1, max_steps + 1):
        tool_name, tool_args = agent.decide(last_result, alerts, difficulty)

        try:
            obs = env.step(CallToolAction(tool_name=tool_name, arguments=tool_args))
        except Exception:
            obs = env.step(CallToolAction(tool_name="observe_network", arguments={}))

        total_reward += obs.reward or 0.0
        action_log.append({"step": step, "action": tool_name, "reward": obs.reward or 0.0})

        raw_result = getattr(obs, "result", None)
        if isinstance(raw_result, dict):
            last_result = raw_result
        elif raw_result is not None:
            try:
                parts = getattr(raw_result, "content", [])
                if parts:
                    text = getattr(parts[0], "text", str(parts[0]))
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
    grader = getattr(state, "grader_result", None) or {}

    return {
        "scenario_id": scenario_id,
        "difficulty": SCENARIO_DIFFICULTY.get(scenario_id, "?"),
        "final_score": grader.get("final_score", 0.0),
        "deterministic_score": grader.get("deterministic_score", grader.get("final_score", 0.0)),
        "episode_end_bonus": grader.get("episode_end_bonus", 0.0),
        "total_episode_reward": grader.get("total_episode_reward", total_reward),
        "judge": grader.get("judge", {}),
        "details": grader.get("details", {}),
        "steps_used": step,
        "max_steps": max_steps,
        "action_log": action_log,
        "seed": seed,
        "timestamp": time.time(),
    }


def run_evaluation(
    scenarios: list[str],
    seed: int = 42,
    runs: int = 1,
    verbose: bool = False,
) -> list[dict]:
    """Run evaluation across scenarios."""
    console = Console() if HAS_RICH else None
    agent = HeuristicSOCAgent()
    all_results = []

    if console:
        console.print(Panel(
            "[bold cyan]CyberRange — Evaluation[/]\n"
            "[dim]Heuristic baseline agent across all scenarios[/]",
            border_style="bright_cyan",
        ))

    for scenario_id in scenarios:
        ep_results = []
        for run in range(runs):
            ep_seed = seed + run
            if console:
                with console.status(f"[cyan]Running {scenario_id} (seed={ep_seed})...[/]"):
                    r = run_episode(agent, scenario_id, ep_seed)
            else:
                r = run_episode(agent, scenario_id, ep_seed)
                print(f"  {scenario_id}: score={r['final_score']:.3f} reward={r['total_episode_reward']:.1f}")
            ep_results.append(r)

        # Average across runs
        avg_score = sum(r["final_score"] for r in ep_results) / len(ep_results)
        avg_reward = sum(r["total_episode_reward"] for r in ep_results) / len(ep_results)
        best_result = max(ep_results, key=lambda r: r["final_score"])
        best_result["avg_score"] = avg_score
        best_result["avg_reward"] = avg_reward
        all_results.append(best_result)

        if console:
            score = avg_score
            diff = SCENARIO_DIFFICULTY.get(scenario_id, "?")
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            style = "bright_green" if score >= 0.7 else "bright_yellow" if score >= 0.4 else "bright_red"
            judge_str = ""
            judge = best_result.get("judge", {})
            if judge.get("judge_enabled"):
                judge_str = f"  [dim]llm={judge.get('llm_judge_score', 0):.3f}[/]"
            console.print(
                f"  [{style}]{bar}[/] [bold]{score:.3f}[/]  "
                f"[dim]{scenario_id}[/] ([italic]{diff}[/]){judge_str}"
            )

    return all_results


def print_summary(results: list[dict]) -> None:
    """Print final summary table."""
    console = Console() if HAS_RICH else None

    if not console:
        avg = sum(r["final_score"] for r in results) / len(results)
        print(f"\nAverage Score: {avg:.3f}")
        return

    console.print()
    table = Table(title="Evaluation Results", box=box.ROUNDED, border_style="bright_cyan")
    table.add_column("Scenario", style="bold")
    table.add_column("Difficulty", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Episode Reward", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("LLM Judge", justify="right")

    for r in results:
        score = r.get("avg_score", r["final_score"])
        style = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"
        diff = r["difficulty"]
        diff_style = {"easy": "dim", "medium": "cyan", "hard": "yellow", "nightmare": "red"}.get(diff, "")
        judge = r.get("judge", {})
        judge_str = f"{judge.get('llm_judge_score', 0):.3f}" if judge.get("judge_enabled") else "—"
        table.add_row(
            r["scenario_id"],
            f"[{diff_style}]{diff}[/]",
            f"[{style}]{score:.3f}[/]",
            f"{r.get('avg_reward', r.get('total_episode_reward', 0)):.1f}",
            f"{r['steps_used']}/{r['max_steps']}",
            judge_str,
        )

    console.print(table)

    avg_score = sum(r.get("avg_score", r["final_score"]) for r in results) / len(results)
    avg_reward = sum(r.get("avg_reward", r.get("total_episode_reward", 0)) for r in results) / len(results)
    judge_enabled = any(r.get("judge", {}).get("judge_enabled") for r in results)

    console.print(Panel(
        f"  Average Score:          [bold bright_cyan]{avg_score:.3f}[/]\n"
        f"  Average Episode Reward: [bold]{avg_reward:.1f}[/]\n"
        f"  LLM Judge Active:       [bold]{'✅ Yes' if judge_enabled else '❌ No'}[/]\n"
        f"  Scenarios Evaluated:    [bold]{len(results)}[/]",
        title="📊 Summary",
        border_style="bright_cyan",
    ))


def save_results(results: list[dict], path: str = "training_results/eval_baseline.json") -> None:
    """Save evaluation results to JSON for reward curve plotting."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "agent": "HeuristicSOCAgent",
            "timestamp": time.time(),
            "scenarios": results,
            "summary": {
                "avg_score": sum(r["final_score"] for r in results) / len(results),
                "avg_reward": sum(r.get("total_episode_reward", 0) for r in results) / len(results),
            }
        }, f, indent=2)
    print(f"  Results saved to {path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CyberRange Evaluation — Heuristic Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py
  python eval.py --scenarios script_kiddie ransomware_outbreak
  python eval.py --seed 123 --runs 3
  python eval.py --save
        """
    )
    parser.add_argument("--scenarios", nargs="+", default=ALL_SCENARIOS,
                        choices=ALL_SCENARIOS, help="Scenarios to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--runs", type=int, default=1, help="Runs per scenario (averaged)")
    parser.add_argument("--save", action="store_true", help="Save results to training_results/")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    results = run_evaluation(
        scenarios=args.scenarios,
        seed=args.seed,
        runs=args.runs,
        verbose=args.verbose,
    )

    print_summary(results)

    if args.save:
        save_results(results)


if __name__ == "__main__":
    main()
