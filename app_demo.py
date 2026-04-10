"""
CyberRange — Live Attack Visualization Dashboard

A Gradio-based interactive dashboard that visualizes cyber attacks
unfolding in real-time on the enterprise network.

This is the HuggingFace Spaces entry point — when judges open the Space,
they see a live, interactive network defense simulation.

Usage:
    python app_demo.py              # Launch locally on port 7860
    gradio app_demo.py              # Launch with hot reload
"""

import json
import time
import html
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from openenv.core.env_server.mcp_types import CallToolAction
from cyber_range.server.cyber_environment import CyberRangeEnvironment


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Network visualization — SVG-based topology map
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NODE_POSITIONS = {
    "fw-01": (400, 60),   "web-01": (550, 60),
    "dc-01": (150, 200),  "mail-01": (300, 200), "app-01": (450, 200),
    "db-01": (600, 200),  "backup-01": (750, 200),
    "ws-01": (150, 340),  "ws-02": (300, 340),
    "ws-03": (450, 340),  "ws-04": (600, 340),
    "honeypot-01": (750, 340),
}

NODE_ICONS = {
    "firewall": "🔥", "domain_controller": "🏛️", "web_server": "🌐",
    "mail_server": "📧", "app_server": "⚙️", "database": "🗄️",
    "workstation": "💻", "backup": "💾", "honeypot": "🍯",
}

STATUS_COLORS = {
    "healthy": "#22c55e", "compromised": "#ef4444", "encrypted": "#7c3aed",
    "isolated": "#3b82f6", "patched": "#06b6d4",
}

# Network connections for topology lines
NETWORK_LINKS = [
    ("fw-01", "web-01"), ("fw-01", "dc-01"), ("fw-01", "mail-01"),
    ("fw-01", "db-01"), ("dc-01", "mail-01"), ("dc-01", "app-01"),
    ("app-01", "db-01"), ("db-01", "backup-01"),
    ("dc-01", "ws-01"), ("dc-01", "ws-02"), ("dc-01", "ws-03"), ("dc-01", "ws-04"),
    ("app-01", "ws-03"),
]


def render_network_svg(nodes_data: list[dict], attack_events: list[str] = None) -> str:
    """Render the enterprise network as an animated SVG."""
    svg_width, svg_height = 900, 430

    # Build node status map
    status_map = {}
    type_map = {}
    for n in nodes_data:
        nid = n.get("node_id", "")
        status_map[nid] = n.get("status", "healthy")
        type_map[nid] = n.get("node_type", "workstation")

    svg = f"""<svg viewBox="0 0 {svg_width} {svg_height}" xmlns="http://www.w3.org/2000/svg"
              style="background:#0f172a;border-radius:12px;font-family:'Inter',sans-serif;">
    <defs>
      <filter id="glow"><feGaussianBlur stdDeviation="3" result="blur"/>
        <feComposite in="SourceGraphic" in2="blur" operator="over"/></filter>
      <filter id="shadow"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.3"/></filter>
      <radialGradient id="pulse-red" cx="50%" cy="50%" r="50%">
        <animate attributeName="r" values="30%;60%;30%" dur="2s" repeatCount="indefinite"/>
        <stop offset="0%" stop-color="#ef4444" stop-opacity="0.6"/>
        <stop offset="100%" stop-color="#ef4444" stop-opacity="0"/></radialGradient>
      <radialGradient id="pulse-green" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#22c55e" stop-opacity="0.3"/>
        <stop offset="100%" stop-color="#22c55e" stop-opacity="0"/></radialGradient>
    </defs>
    <text x="16" y="28" fill="#94a3b8" font-size="13" font-weight="600">
      ENTERPRISE NETWORK TOPOLOGY — LIVE</text>"""

    # Draw network links
    for src, dst in NETWORK_LINKS:
        if src in NODE_POSITIONS and dst in NODE_POSITIONS:
            x1, y1 = NODE_POSITIONS[src]
            x2, y2 = NODE_POSITIONS[dst]
            # Color link red if either node is compromised
            s1 = status_map.get(src, "healthy")
            s2 = status_map.get(dst, "healthy")
            is_threat = s1 in ("compromised", "encrypted") or s2 in ("compromised", "encrypted")
            color = "#ef4444" if is_threat else "#1e293b"
            opacity = "0.7" if is_threat else "0.4"
            width = "2" if is_threat else "1"
            svg += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>'

    # Draw nodes
    for nid, (x, y) in NODE_POSITIONS.items():
        status = status_map.get(nid, "healthy")
        ntype = type_map.get(nid, "workstation")
        color = STATUS_COLORS.get(status, "#22c55e")
        icon = NODE_ICONS.get(ntype, "💻")

        # Pulse effect for compromised/encrypted nodes
        if status in ("compromised", "encrypted"):
            svg += f'<circle cx="{x}" cy="{y}" r="28" fill="url(#pulse-red)"/>'

        # Node circle
        svg += f"""<circle cx="{x}" cy="{y}" r="22" fill="{color}" opacity="0.15"
                    stroke="{color}" stroke-width="2" filter="url(#shadow)"/>"""
        svg += f"""<circle cx="{x}" cy="{y}" r="18" fill="#0f172a"
                    stroke="{color}" stroke-width="1.5"/>"""

        # Status indicator dot
        svg += f'<circle cx="{x+15}" cy="{y-15}" r="5" fill="{color}" filter="url(#glow)"/>'

        # Icon
        svg += f'<text x="{x}" y="{y+5}" text-anchor="middle" font-size="16">{icon}</text>'

        # Label
        label_color = "#f8fafc" if status != "healthy" else "#94a3b8"
        svg += f'<text x="{x}" y="{y+38}" text-anchor="middle" fill="{label_color}" font-size="10" font-weight="500">{nid}</text>'

        # Status badge
        if status != "healthy":
            badge_text = status.upper()
            badge_w = len(badge_text) * 6 + 10
            svg += f"""<rect x="{x - badge_w/2}" y="{y+42}" width="{badge_w}" height="14"
                        rx="4" fill="{color}" opacity="0.9"/>"""
            svg += f'<text x="{x}" y="{y+53}" text-anchor="middle" fill="white" font-size="8" font-weight="700">{badge_text}</text>'

    # Legend
    legend_y = svg_height - 30
    legends = [("Healthy", "#22c55e"), ("Compromised", "#ef4444"),
               ("Encrypted", "#7c3aed"), ("Isolated", "#3b82f6"), ("Patched", "#06b6d4")]
    for i, (label, color) in enumerate(legends):
        lx = 50 + i * 150
        svg += f'<circle cx="{lx}" cy="{legend_y}" r="5" fill="{color}"/>'
        svg += f'<text x="{lx+10}" y="{legend_y+4}" fill="#94a3b8" font-size="10">{label}</text>'

    svg += "</svg>"
    return svg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MITRE ATT&CK heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MITRE_TECHNIQUES = {
    "T1110.001": ("Brute Force", "Credential Access"),
    "T1190": ("Exploit Public App", "Initial Access"),
    "T1566.001": ("Spearphishing", "Initial Access"),
    "T1204.002": ("Malicious File", "Execution"),
    "T1003.001": ("LSASS Memory", "Credential Access"),
    "T1021.002": ("SMB Shares", "Lateral Movement"),
    "T1078.002": ("Domain Accounts", "Privilege Escalation"),
    "T1074.001": ("Local Staging", "Collection"),
    "T1041": ("Exfil Over C2", "Exfiltration"),
    "T1567.002": ("Cloud Storage", "Exfiltration"),
    "T1486": ("Data Encryption", "Impact"),
    "T1490": ("Inhibit Recovery", "Impact"),
    "T1489": ("Service Stop", "Impact"),
    "T1195.002": ("Supply Chain", "Initial Access"),
    "T1059.001": ("PowerShell", "Execution"),
    "T1105": ("Tool Transfer", "Command & Control"),
}


def render_mitre_html(active_techniques: list[str] = None) -> str:
    """Render MITRE ATT&CK heatmap as HTML."""
    active = set(active_techniques or [])

    html_out = """<div style="background:#0f172a;border-radius:12px;padding:16px;font-family:'Inter',sans-serif;">
    <h3 style="color:#f8fafc;margin:0 0 12px 0;font-size:14px;">
      🎯 MITRE ATT&CK Coverage</h3>
    <div style="display:flex;flex-wrap:wrap;gap:6px;">"""

    for tid, (name, tactic) in MITRE_TECHNIQUES.items():
        is_active = tid in active
        bg = "#dc2626" if is_active else "#1e293b"
        border = "#ef4444" if is_active else "#334155"
        text_color = "#fecaca" if is_active else "#64748b"
        glow = "box-shadow:0 0 8px rgba(239,68,68,0.5);" if is_active else ""

        html_out += f"""<div style="background:{bg};border:1px solid {border};border-radius:6px;
            padding:4px 8px;{glow}">
            <div style="color:{'#fef2f2' if is_active else '#94a3b8'};font-size:10px;font-weight:700;">{tid}</div>
            <div style="color:{text_color};font-size:9px;">{name}</div>
            <div style="color:{'#fca5a5' if is_active else '#475569'};font-size:8px;font-style:italic;">{tactic}</div>
        </div>"""

    html_out += "</div></div>"
    return html_out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Smart Agent for Demo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DemoAgent:
    """Expert SOC analyst agent for the live demo."""

    def __init__(self):
        self.step = 0
        self.investigated = set()
        self.blocked = set()
        self.dismissed = set()
        self.isolated = set()
        self.patched = set()
        self.honeypot = False
        self.fps = []
        self.threats = []
        self.compromised = []
        self.ext_ips = []

    def reset(self):
        self.__init__()

    def act(self, obs_data, alerts, difficulty="easy"):
        self.step += 1

        if self.step == 1:
            return "observe_network", {}, "🔍 ASSESS: Initiating situational awareness scan of entire network."

        # Deploy honeypot early for hard+ scenarios
        if not self.honeypot and difficulty in ("hard", "nightmare") and self.step <= 3:
            self.honeypot = True
            return "deploy_honeypot", {}, "🍯 PLAN: Deploying honeypot to gather adversary intelligence."

        # Process last evidence
        if isinstance(obs_data, dict):
            det = obs_data.get("details", {})
            if isinstance(det, dict):
                ev = det.get("forensic_evidence", "").lower()
                aid = det.get("alert_id", "")
                if ev:
                    if any(w in ev for w in ["benign", "routine", "scheduled", "legitimate", "baseline"]):
                        if aid:
                            self.fps.append(aid)
                    else:
                        src = det.get("source_ip", "")
                        node = det.get("related_node_id", "") or det.get("related_node", "")
                        if src and not src.startswith("10.0."):
                            self.ext_ips.append(src)
                        if node:
                            self.compromised.append(node)

        # Priority 1: Investigate high/critical alerts
        sorted_alerts = sorted(alerts, key=lambda a: {
            "critical": 0, "high": 1, "medium": 2, "low": 3
        }.get(a.get("severity", "low"), 4))

        for a in sorted_alerts:
            aid = a.get("alert_id", "")
            if aid and aid not in self.investigated:
                self.investigated.add(aid)
                sev = a.get("severity", "?")
                return "investigate_alert", {"alert_id": aid}, \
                    f"🔍 PRIORITIZE: Investigating [{sev.upper()}] alert {aid}."

        # Priority 2: Block external attacker IPs
        for ip in list(self.ext_ips):
            if ip not in self.blocked:
                self.blocked.add(ip)
                self.ext_ips.remove(ip)
                return "block_ip", {"ip_address": ip}, \
                    f"🛡️ ACT: Blocking attacker IP {ip} at firewall."

        for ip in ["185.220.101.42", "94.232.46.19", "45.155.205.233"]:
            if ip not in self.blocked:
                self.blocked.add(ip)
                return "block_ip", {"ip_address": ip}, \
                    f"🛡️ ACT: Proactively blocking known-malicious IP {ip}."

        # Priority 3: Dismiss confirmed FPs
        for aid in list(self.fps):
            if aid not in self.dismissed:
                self.dismissed.add(aid)
                self.fps.remove(aid)
                return "dismiss_alert", {"alert_id": aid}, \
                    f"✅ CONSIDER: Evidence shows benign activity. Dismissing {aid}."

        # Priority 4: Isolate compromised hosts
        for nid in list(self.compromised):
            if nid not in self.isolated:
                self.isolated.add(nid)
                self.compromised.remove(nid)
                return "isolate_host", {"node_id": nid}, \
                    f"🔒 ACT: Isolating confirmed-compromised host {nid}."

        return "observe_network", {}, "🔄 ASSESS: Scanning for new threats."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Simulation Runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCENARIOS = {
    "script_kiddie": ("🟢 Script Kiddie Brute Force", "easy"),
    "phishing_campaign": ("🟡 Phishing Campaign Triage", "medium"),
    "apt_lateral_movement": ("🔴 APT Kill Chain", "hard"),
    "ransomware_outbreak": ("🔴 Ransomware Outbreak", "hard"),
    "supply_chain_compromise": ("🔴 Supply Chain Attack", "hard"),
    "insider_threat_apt": ("💀 Insider + External APT", "nightmare"),
}


def run_simulation(scenario_id, speed):
    """Generator that yields simulation state at each step."""
    env = CyberRangeEnvironment()
    obs = env.reset(task_id=scenario_id, seed=42)
    metadata = obs.metadata or {}
    scenario = metadata.get("scenario", {})
    max_steps = scenario.get("max_steps", 20)
    difficulty = scenario.get("difficulty", "easy")
    alerts = metadata.get("pending_alerts", [])
    topo = metadata.get("network_topology", [])

    agent = DemoAgent()
    last_result = metadata
    event_log = []
    all_mitre = []

    # Get MITRE techniques from scenario
    mitre_techs = scenario.get("mitre_techniques_covered", [])
    all_mitre = list(mitre_techs)

    for step in range(1, max_steps + 1):
        tool, args, reasoning = agent.act(last_result, alerts, difficulty)

        try:
            obs = env.step(CallToolAction(tool_name=tool, arguments=args))
        except Exception:
            obs = env.step(CallToolAction(tool_name="observe_network", arguments={}))

        reward = obs.reward or 0.0

        # Parse result
        raw = getattr(obs, "result", None)
        if isinstance(raw, dict):
            last_result = raw
        elif raw is not None:
            try:
                cp = getattr(raw, "content", [])
                if cp:
                    t = getattr(cp[0], "text", str(cp[0]))
                    try:
                        last_result = json.loads(t)
                    except (json.JSONDecodeError, TypeError):
                        last_result = {}
                else:
                    last_result = {}
            except Exception:
                last_result = {}
        else:
            last_result = {}

        # Update alerts
        if isinstance(last_result, dict) and "pending_alerts" in last_result:
            alerts = last_result["pending_alerts"]

        # Update topology
        if isinstance(last_result, dict) and "network_topology" in last_result:
            topo = last_result["network_topology"]

        # Build event log entry
        reward_str = f"+{reward:.1f}" if reward >= 0 else f"{reward:.1f}"
        action_desc = f"`{tool}({json.dumps(args)[:60]})`" if args else f"`{tool}()`"
        event_log.append({
            "step": step,
            "action": action_desc,
            "reasoning": reasoning,
            "reward": reward_str,
        })

        # Yield current state
        yield {
            "step": step,
            "max_steps": max_steps,
            "topo": topo,
            "event_log": event_log[-8:],  # Last 8 events
            "alerts": alerts,
            "mitre": all_mitre,
            "done": obs.done,
            "reward": reward,
        }

        if obs.done:
            break

    # Final grading
    state = env.state
    grader = getattr(state, "grader_result", None) or {}
    yield {
        "step": max_steps,
        "max_steps": max_steps,
        "topo": topo,
        "event_log": event_log[-8:],
        "alerts": [],
        "mitre": all_mitre,
        "done": True,
        "grader": grader,
        "reward": 0,
    }


def format_event_log_html(events):
    """Format event log as styled HTML."""
    if not events:
        return "<div style='color:#64748b;padding:12px;'>Waiting...</div>"

    html_out = """<div style="background:#0f172a;border-radius:12px;padding:12px;
        font-family:'JetBrains Mono','Fira Code',monospace;font-size:12px;">"""

    for e in events:
        rwd = e["reward"]
        rwd_color = "#22c55e" if rwd.startswith("+") else "#ef4444"
        reasoning_escaped = html.escape(e["reasoning"])
        action_escaped = html.escape(e["action"])

        html_out += f"""<div style="border-bottom:1px solid #1e293b;padding:6px 0;">
            <span style="color:#64748b;">Step {e['step']}</span>
            <span style="color:{rwd_color};float:right;font-weight:700;">{rwd}</span><br/>
            <span style="color:#f8fafc;">{action_escaped}</span><br/>
            <span style="color:#94a3b8;font-style:italic;font-size:11px;">{reasoning_escaped}</span>
        </div>"""

    html_out += "</div>"
    return html_out


def format_score_html(grader):
    """Format final score as styled HTML."""
    if not grader:
        return ""

    score = grader.get("final_score", 0)
    details = grader.get("details", {})
    score_pct = int(score * 100)

    # Score color
    if score >= 0.8:
        color, label = "#22c55e", "EXCELLENT"
    elif score >= 0.6:
        color, label = "#eab308", "GOOD"
    elif score >= 0.4:
        color, label = "#f97316", "FAIR"
    else:
        color, label = "#ef4444", "NEEDS IMPROVEMENT"

    bar_width = max(5, score_pct)

    h = f"""<div style="background:#0f172a;border-radius:12px;padding:20px;text-align:center;font-family:'Inter',sans-serif;">
        <div style="font-size:48px;font-weight:800;color:{color};">{score:.3f}</div>
        <div style="font-size:14px;color:{color};font-weight:600;margin-bottom:12px;">{label}</div>
        <div style="background:#1e293b;border-radius:8px;height:12px;margin:8px 0;">
            <div style="background:{color};border-radius:8px;height:12px;width:{bar_width}%;
                 transition:width 1s ease;"></div></div>
        <div style="display:flex;justify-content:space-around;margin-top:16px;flex-wrap:wrap;">"""

    components = [
        ("Threats", grader.get("threat_response", 0), 0.35),
        ("False Pos.", grader.get("false_positive_handling", 0), 0.20),
        ("Data Prot.", grader.get("data_protection", 0), 0.20),
        ("Collateral", grader.get("collateral_damage", 0), 0.15),
        ("Efficiency", grader.get("efficiency", 0), 0.10),
    ]

    for name, val, weight in components:
        normalized = val / weight if weight > 0 else 0
        comp_color = "#22c55e" if normalized >= 0.7 else "#eab308" if normalized >= 0.4 else "#ef4444"
        h += f"""<div style="text-align:center;padding:4px 8px;">
            <div style="color:{comp_color};font-size:18px;font-weight:700;">{val:.3f}</div>
            <div style="color:#64748b;font-size:10px;">{name} ({int(weight*100)}%)</div></div>"""

    h += f"""</div>
        <div style="color:#475569;font-size:11px;margin-top:12px;">
            {details.get('threats_neutralized', '?')} threats • 
            {details.get('steps_used', '?')} steps •
            {details.get('data_exfiltrated_mb', 0)} MB exfiltrated •
            Adversary: {details.get('adversary_behavior', '?')}
        </div></div>"""

    return h


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gradio App
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_demo(scenario_choice, speed_choice):
    """Run the full simulation and return final state."""
    # Map display names to IDs
    scenario_id = None
    for sid, (name, _) in SCENARIOS.items():
        if name == scenario_choice:
            scenario_id = sid
            break
    if not scenario_id:
        scenario_id = "script_kiddie"

    speed = {"Fast": 0, "Normal": 0.3, "Slow": 0.8}.get(speed_choice, 0.3)

    # Run simulation to completion
    last_state = None
    for state in run_simulation(scenario_id, speed):
        last_state = state

    if not last_state:
        return "<div>Error</div>", "<div>Error</div>", "<div>Error</div>", ""

    topo = last_state.get("topo", [])
    events = last_state.get("event_log", [])
    mitre = last_state.get("mitre", [])
    grader = last_state.get("grader", {})

    network_html = render_network_svg(topo)
    event_html = format_event_log_html(events)
    mitre_html = render_mitre_html(mitre)
    score_html = format_score_html(grader) if grader else ""

    return network_html, event_html, mitre_html, score_html


# Build the Gradio interface
CSS = """
.gradio-container { max-width: 1200px !important; }
.dark { background: #0a0a0a !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="CyberRange — SOC Analyst RL Environment",
    css=CSS,
    theme=gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
) as demo:
    gr.Markdown("""
    # 🛡️ CyberRange — Live Attack Visualization
    **Adaptive adversaries. MITRE ATT&CK aligned. Multi-objective grading.**

    Watch an AI SOC analyst defend an enterprise network against cyber attacks in real-time.
    The adversary adapts: rotating C2 IPs when blocked, persisting through patching, and deploying decoys.
    """)

    with gr.Row():
        scenario_dd = gr.Dropdown(
            choices=[name for name, _ in SCENARIOS.values()],
            value="🟢 Script Kiddie Brute Force",
            label="Attack Scenario",
        )
        speed_dd = gr.Dropdown(
            choices=["Fast", "Normal", "Slow"],
            value="Fast",
            label="Simulation Speed",
        )
        run_btn = gr.Button("▶️ Launch Attack", variant="primary", size="lg")

    with gr.Row():
        with gr.Column(scale=3):
            network_view = gr.HTML(
                value=render_network_svg([
                    {"node_id": nid, "status": "healthy", "node_type": "workstation"}
                    for nid in NODE_POSITIONS
                ]),
                label="Network Topology",
            )
        with gr.Column(scale=2):
            event_view = gr.HTML(
                value="<div style='background:#0f172a;border-radius:12px;padding:24px;color:#64748b;text-align:center;font-family:Inter,sans-serif;'>Select a scenario and click Launch Attack</div>",
                label="Agent Actions",
            )

    with gr.Row():
        with gr.Column(scale=3):
            mitre_view = gr.HTML(value=render_mitre_html(), label="MITRE Coverage")
        with gr.Column(scale=2):
            score_view = gr.HTML(
                value="<div style='background:#0f172a;border-radius:12px;padding:24px;color:#64748b;text-align:center;font-family:Inter,sans-serif;'>Score will appear after simulation</div>",
                label="Score",
            )

    run_btn.click(
        fn=run_demo,
        inputs=[scenario_dd, speed_dd],
        outputs=[network_view, event_view, mitre_view, score_view],
    )

    gr.Markdown("""
    ---
    **Built for the [Meta PyTorch × Scaler OpenEnv Hackathon](https://unstop.com)** |
    [GitHub](https://github.com/keshav-005/cyber_range) |
    [OpenEnv 0.2.2](https://github.com/meta-pytorch/openenv) |
    6 scenarios · 16 techniques · Adaptive adversary · Seed: 42
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
