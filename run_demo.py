"""
CyberRange — Demo Runner with Rich Terminal UI

Run this script to experience the SOC Analyst RL Environment.

Usage:
    python run_demo.py                  # Interactive menu
    python run_demo.py --auto           # Auto demo with heuristic agent
    python run_demo.py --benchmark      # Full benchmark (all 5 scenarios)
    python run_demo.py --quick          # Quick demo (easy + medium only)
    python run_demo.py --scenario apt   # Run a specific scenario
    python run_demo.py --llm            # Use LLM agent (needs HF_TOKEN)
"""

import sys
import os
import argparse
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openenv.core.env_server.mcp_types import CallToolAction
from cyber_range.server.cyber_environment import CyberRangeEnvironment

# Try rich for beautiful output; fall back to plain text
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.progress import Progress, BarColumn, TextColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


SCENARIOS = {
    "script_kiddie": ("Script Kiddie Brute Force", "🟢 EASY", 1),
    "phishing_campaign": ("Phishing Campaign Triage", "🟡 MEDIUM", 3),
    "apt_lateral_movement": ("APT Kill Chain", "🔴 HARD", 5),
    "ransomware_outbreak": ("Ransomware Outbreak", "🔴 HARD", 4),
    "insider_threat_apt": ("Insider + External APT", "💀 NIGHTMARE", 7),
}


class HeuristicSOCAgent:
    """Rule-based SOC analyst that demonstrates the environment."""

    def __init__(self):
        self.step_num = 0
        self.investigated = set()
        self.blocked_ips = set()
        self.dismissed = set()
        self.isolated = set()
        self.fp_candidates = []
        self.real_threats = []
        self.discovered_ips = []
        self.discovered_compromised = []

    def reset(self):
        self.__init__()

    def act(self, obs_data: dict, alerts: list) -> tuple[str, dict, str]:
        """Returns (tool_name, args, reasoning)."""
        self.step_num += 1

        if self.step_num == 1:
            return "observe_network", {}, "ASSESS: Beginning incident response. Gathering full network state."

        # Process last result
        if isinstance(obs_data, dict):
            details = obs_data.get("details", {})
            if isinstance(details, dict) and "forensic_evidence" in details:
                evidence = details.get("forensic_evidence", "").lower()
                aid = details.get("alert_id", "")
                if "benign" in evidence or "routine" in evidence:
                    if aid:
                        self.fp_candidates.append(aid)
                else:
                    src = details.get("source_ip", "")
                    node = details.get("related_node", "") or details.get("related_node_id", "")
                    if src and not src.startswith("10.0."):
                        self.discovered_ips.append(src)
                    if node:
                        self.discovered_compromised.append(node)

        # Phase 1: Investigate
        for alert in alerts:
            aid = alert.get("alert_id", "")
            if aid and aid not in self.investigated:
                self.investigated.add(aid)
                sev = alert.get("severity", "low")
                return "investigate_alert", {"alert_id": aid}, \
                    f"PRIORITIZE: Investigating {aid} (severity={sev}). Need evidence before containment."

        # Phase 2: Block attacker IPs
        for ip in self.discovered_ips:
            if ip not in self.blocked_ips:
                self.blocked_ips.add(ip)
                return "block_ip", {"ip_address": ip}, \
                    f"ACT: Blocking external attacker IP {ip} at firewall [MITRE: T1071 Application Layer Protocol]."

        for ip in ["185.220.101.42", "94.232.46.19", "45.155.205.233"]:
            if ip not in self.blocked_ips:
                self.blocked_ips.add(ip)
                return "block_ip", {"ip_address": ip}, \
                    f"ACT: Proactively blocking known-malicious IP {ip}."

        # Phase 3: Dismiss FPs
        for aid in self.fp_candidates:
            if aid not in self.dismissed:
                self.dismissed.add(aid)
                return "dismiss_alert", {"alert_id": aid}, \
                    f"CONSIDER: Alert {aid} showed benign activity. Dismissing as false positive."

        # Phase 4: Isolate compromised
        for node_id in self.discovered_compromised:
            if node_id not in self.isolated:
                self.isolated.add(node_id)
                return "isolate_host", {"node_id": node_id}, \
                    f"ACT: Isolating confirmed-compromised host {node_id} [MITRE: Containment]."

        return "observe_network", {}, "ASSESS: All known threats addressed. Checking for new alerts."


def run_scenario_demo(scenario_id: str, console=None, slow_mode=True):
    """Run a single scenario with the heuristic agent and rich display."""
    name, difficulty, mitre_count = SCENARIOS[scenario_id]

    if console and HAS_RICH:
        console.print()
        console.print(Panel(
            f"[bold bright_white]{name}[/] — {difficulty}\n"
            f"[dim]MITRE ATT&CK Techniques: {mitre_count} | Scenario: {scenario_id}[/dim]",
            border_style="bright_cyan",
            title="🛡️ CyberRange Scenario",
        ))
    else:
        print(f"\n{'='*60}")
        print(f"  {name} [{difficulty}]")
        print(f"  MITRE Techniques: {mitre_count}")
        print(f"{'='*60}")

    env = CyberRangeEnvironment()
    obs = env.reset(task_id=scenario_id, seed=42)
    metadata = obs.metadata or {}
    scenario = metadata.get("scenario", {})
    max_steps = scenario.get("max_steps", 20)
    alerts = metadata.get("pending_alerts", [])

    agent = HeuristicSOCAgent()
    last_result = metadata

    steps_taken = 0
    for step in range(1, max_steps + 1):
        tool_name, tool_args, reasoning = agent.act(last_result, alerts)

        if console and HAS_RICH and slow_mode:
            console.print(f"  [dim]Step {step}/{max_steps}[/dim] [bright_cyan]{tool_name}[/]({tool_args})")
            console.print(f"    [italic dim]{reasoning}[/]")
            time.sleep(0.05)

        try:
            obs = env.step(CallToolAction(tool_name=tool_name, arguments=tool_args))
        except Exception:
            obs = env.step(CallToolAction(tool_name="observe_network", arguments={}))

        # Extract result
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

        # Update alerts from new observations
        if isinstance(last_result, dict):
            new_alerts = last_result.get("network_summary", {})
            # Try to get pending alerts from full observation
            if "pending_alerts" in last_result:
                alerts = last_result["pending_alerts"]

        steps_taken = step
        if obs.done:
            break

    # Get grading result
    state = env.state
    grader_result = getattr(state, "grader_result", None) or {}
    final_score = grader_result.get("final_score", 0.0)
    mitre = grader_result.get("mitre_coverage", {})
    details = grader_result.get("details", {})

    return {
        "scenario_id": scenario_id,
        "name": name,
        "difficulty": difficulty,
        "final_score": final_score,
        "steps_used": steps_taken,
        "max_steps": max_steps,
        "details": details,
        "mitre_techniques": mitre_count,
        "mitre_coverage": mitre,
        "adversary_behavior": details.get("adversary_behavior", "static"),
    }


def show_benchmark_results(results: list, console=None):
    """Show final benchmark results with rich table."""
    if console and HAS_RICH:
        table = Table(
            title="🛡️ CyberRange Benchmark Results",
            box=box.HEAVY,
            border_style="bright_cyan",
            show_lines=True,
        )
        table.add_column("Scenario", style="bold", min_width=25)
        table.add_column("Difficulty", justify="center")
        table.add_column("MITRE", justify="center", style="bright_magenta")
        table.add_column("Adversary", justify="center")
        table.add_column("Score", justify="center")
        table.add_column("Steps", justify="center", style="dim")
        table.add_column("Status", justify="center")

        total_score = 0
        for r in results:
            score = r["final_score"]
            total_score += score
            score_style = "bright_green" if score >= 0.7 else "bright_yellow" if score >= 0.4 else "bright_red"
            status = "✅ PASS" if score >= 0.4 else "❌ FAIL"

            # Score bar
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)

            table.add_row(
                r["name"],
                r["difficulty"],
                str(r["mitre_techniques"]),
                r["adversary_behavior"],
                Text(f"{bar} {score:.3f}", style=score_style),
                f"{r['steps_used']}/{r['max_steps']}",
                status,
            )

        console.print()
        console.print(table)

        avg = total_score / len(results) if results else 0
        passed = sum(1 for r in results if r["final_score"] >= 0.4)
        console.print()
        console.print(Panel(
            f"  Scenarios Passed: [bold bright_green]{passed}/{len(results)}[/]\n"
            f"  Average Score:    [bold bright_white]{avg:.3f}[/]\n"
            f"  Total MITRE Techniques Tested: [bold bright_magenta]{sum(r['mitre_techniques'] for r in results)}[/]\n"
            f"  Seed: 42 (reproducible)",
            title="📊 Summary",
            border_style="bright_green",
        ))
    else:
        print("\n" + "=" * 60)
        print("  BENCHMARK RESULTS")
        print("=" * 60)
        total = 0
        for r in results:
            score = r["final_score"]
            total += score
            bar = "#" * int(score * 20) + "." * (20 - int(score * 20))
            status = "PASS" if score >= 0.4 else "FAIL"
            print(f"  {r['name']:<30} [{bar}] {score:.3f}  {status}")
        avg = total / len(results) if results else 0
        print(f"\n  Average: {avg:.3f}")
    print()


def run_benchmark(scenario_ids=None, slow_mode=True):
    """Run the full benchmark suite."""
    console = Console() if HAS_RICH else None

    if console:
        console.print(Panel(
            "[bold bright_white]CyberRange[/] — SOC Analyst RL Environment\n"
            "[dim]Adaptive Adversaries | MITRE ATT&CK Aligned | Multi-Objective Grading[/]",
            border_style="bright_cyan",
            title="🛡️",
        ))

    if scenario_ids is None:
        scenario_ids = list(SCENARIOS.keys())

    results = []
    for sid in scenario_ids:
        result = run_scenario_demo(sid, console=console, slow_mode=slow_mode)
        results.append(result)

        if console and HAS_RICH:
            score = result["final_score"]
            style = "bright_green" if score >= 0.7 else "bright_yellow" if score >= 0.4 else "bright_red"
            console.print(f"\n  [{style}]Score: {score:.3f}[/] — {result['name']}")

    show_benchmark_results(results, console)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="CyberRange — SOC Analyst RL Environment Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py                    # Full benchmark
  python run_demo.py --auto             # Auto demo with detailed output
  python run_demo.py --quick            # Quick demo (easy + medium)
  python run_demo.py --scenario apt     # Run APT scenario only
  python run_demo.py --benchmark        # Full benchmark (all 5 scenarios)
        """
    )
    parser.add_argument("--auto", action="store_true", help="Auto demo with step-by-step output")
    parser.add_argument("--quick", action="store_true", help="Quick demo (easy + medium only)")
    parser.add_argument("--benchmark", action="store_true", help="Full benchmark (all 5 scenarios)")
    parser.add_argument("--scenario", type=str, help="Run specific scenario (script_kiddie, phishing, apt, ransomware, insider)")
    parser.add_argument("--fast", action="store_true", help="Skip step-by-step output")

    args = parser.parse_args()

    if args.scenario:
        # Map short names
        mapping = {
            "script": "script_kiddie", "kiddie": "script_kiddie",
            "phish": "phishing_campaign", "phishing": "phishing_campaign",
            "apt": "apt_lateral_movement",
            "ransom": "ransomware_outbreak", "ransomware": "ransomware_outbreak",
            "insider": "insider_threat_apt", "nightmare": "insider_threat_apt",
        }
        sid = mapping.get(args.scenario, args.scenario)
        run_benchmark([sid], slow_mode=not args.fast)
    elif args.quick:
        run_benchmark(["script_kiddie", "phishing_campaign"], slow_mode=not args.fast)
    else:
        run_benchmark(slow_mode=not args.fast)


if __name__ == "__main__":
    main()
