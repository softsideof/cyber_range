"""
CyberRange — Rich Terminal Display Module

Provides formatted, color-coded output for attack scenarios,
forensic evidence, and benchmark results.
"""

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class Display:
    """Rich terminal display for CyberRange demos and benchmarks."""

    def __init__(self, slow_mode: bool = False, delay: float = 0.1):
        self.console = Console() if HAS_RICH else None
        self.slow_mode = slow_mode
        self.delay = delay

    def show_banner(self):
        """Show the CyberRange banner."""
        if not self.console:
            print("\n  🛡️  CyberRange — SOC Analyst RL Environment\n")
            return

        self.console.print(Panel(
            "[bold bright_white]CyberRange[/]\n"
            "[dim]SOC Analyst RL Environment[/]\n\n"
            "[bright_cyan]Adaptive Adversaries[/] · "
            "[bright_magenta]MITRE ATT&CK[/] · "
            "[bright_green]Multi-Objective Grading[/]",
            border_style="bright_cyan",
            title="🛡️",
            subtitle="[dim]OpenEnv 0.2.2[/dim]",
        ))

    def show_forensic_report(self, findings: dict):
        """Display a forensic report as a rich tree with process trees, IOCs, etc."""
        if not self.console or not HAS_RICH:
            import json
            print(json.dumps(findings, indent=2, default=str))
            return

        node_id = findings.get("node_id", "unknown")
        hostname = findings.get("hostname", "unknown")
        risk = findings.get("risk_score", 0)
        risk_style = "bright_red" if risk > 70 else "bright_yellow" if risk > 30 else "bright_green"

        tree = Tree(
            f"[bold]🔍 Forensic Report: {node_id} ({hostname})[/] "
            f"[{risk_style}]Risk: {risk}/100[/]"
        )

        # Process tree
        processes = findings.get("process_tree", [])
        if processes:
            proc_branch = tree.add("[bold bright_cyan]Process Tree[/]")
            for p in processes:
                style = "bright_red bold" if p.get("suspicious") else "dim"
                icon = "⚠️ " if p.get("suspicious") else "  "
                node_text = f"{icon}PID {p['pid']} → {p['name']} [{p.get('user', '?')}]"
                proc_node = proc_branch.add(f"[{style}]{node_text}[/]")
                if p.get("cmd"):
                    proc_node.add(f"[dim]{p['cmd']}[/]")
                if p.get("note"):
                    proc_node.add(f"[bright_red]{p['note']}[/]")

        # Network connections
        connections = findings.get("network_connections", [])
        if connections:
            net_branch = tree.add("[bold bright_yellow]Network Connections[/]")
            for c in connections:
                state_style = "bright_red" if c["state"] == "ESTABLISHED" else "dim"
                net_branch.add(
                    f"[{state_style}]{c['local_addr']} → {c['remote_addr']} "
                    f"({c['state']}) {c.get('note', '')}[/]"
                )

        # File artifacts
        files = findings.get("file_artifacts", [])
        if files:
            file_branch = tree.add("[bold bright_magenta]File Artifacts[/]")
            for f in files:
                file_node = file_branch.add(f"[bright_red]{f['path']}[/]")
                file_node.add(f"[dim]SHA256: {f['sha256'][:16]}... | VT: {f.get('vt_detection', 'N/A')}[/]")
                if f.get("note"):
                    file_node.add(f"[bright_yellow]{f['note']}[/]")

        # Memory indicators
        memory = findings.get("memory_indicators", [])
        if memory:
            mem_branch = tree.add("[bold bright_red]Memory Indicators[/]")
            for m in memory:
                mem_branch.add(f"[bright_red]{m}[/]" if "malicious" not in m.lower() and "no " not in m.lower()
                              else f"[dim]{m}[/]")

        self.console.print()
        self.console.print(tree)
        self.console.print()

    def show_alert_investigation(self, details: dict):
        """Display an alert investigation result."""
        if not self.console:
            print(f"  Alert: {details.get('alert_id')} — {details.get('forensic_evidence', '')[:200]}")
            return

        alert_id = details.get("alert_id", "?")
        severity = details.get("severity", "unknown")
        sev_style = {"critical": "bright_red", "high": "red", "medium": "yellow", "low": "green"}.get(severity, "white")

        ioc = details.get("ioc_summary", {})
        mal = ioc.get("malicious_indicators", 0)
        verdict_style = "bright_red bold" if mal > 0 else "bright_green bold"
        verdict = "🔴 MALICIOUS" if mal > 0 else "🟢 BENIGN"

        self.console.print(Panel(
            f"[bold]Alert:[/] {alert_id} [{sev_style}]{severity.upper()}[/]\n"
            f"[bold]Type:[/] {details.get('alert_type', '?')}\n"
            f"[bold]Source:[/] {details.get('source_ip', '?')} → {details.get('destination_ip', '?')}\n"
            f"[bold]Verdict:[/] [{verdict_style}]{verdict}[/]\n\n"
            f"[dim]{details.get('forensic_evidence', 'No evidence available.')[:500]}[/]",
            title=f"🔍 Investigation: {alert_id}",
            border_style=sev_style,
        ))

    def show_phase_header(self, title: str):
        if self.console:
            self.console.rule(f"[bold bright_cyan]{title}[/]")
        else:
            print(f"\n--- {title} ---")

    def show_success(self, msg: str):
        if self.console:
            self.console.print(f"  [bright_green]✅ {msg}[/]")
        else:
            print(f"  ✅ {msg}")

    def show_warning(self, msg: str):
        if self.console:
            self.console.print(f"  [bright_yellow]⚠️  {msg}[/]")
        else:
            print(f"  ⚠️  {msg}")

    def show_error(self, msg: str):
        if self.console:
            self.console.print(f"  [bright_red]❌ {msg}[/]")
        else:
            print(f"  ❌ {msg}")

    def show_info(self, msg: str, style: str = "dim"):
        if self.console:
            self.console.print(f"  [{style}]{msg}[/]")
        else:
            print(f"  {msg}")
