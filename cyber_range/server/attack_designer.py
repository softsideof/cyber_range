"""
AttackDesigner — LLM-Powered Adversarial Attack Scenario Generator.

Analyzes agent failure patterns from past episodes and generates
novel attack scenarios that specifically target the agent's blind spots.

Inspired by: Kube SRE Gym (Gold, OpenEnv SF Hackathon 2026).
Key innovation: The environment gets harder as the agent improves —
unlimited unique training scenarios without manual engineering.

Usage:
    designer = AttackDesigner()
    failure_logs = [
        {"scenario_id": "ransomware_outbreak", "score": 0.3, "actions": [...], "failure_reason": "missed FP dismissal"},
    ]
    new_scenario = designer.design_scenario(failure_logs, target_difficulty="hard")
    # Returns ScenarioConfig ready to register with AttackEngine
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

try:
    from ..models import (
        AlertType, AttackPhase, Difficulty, ScenarioConfig,
    )
except ImportError:
    from cyber_range.models import (
        AlertType, AttackPhase, Difficulty, ScenarioConfig,
    )


# ─────────────────────────────────────────────────────────────
# Designer prompt — the core LLM instruction set
# ─────────────────────────────────────────────────────────────

DESIGNER_SYSTEM_PROMPT = """You are an elite red team operator and cybersecurity curriculum designer.
Your job is to design realistic, multi-stage attack scenarios for training AI SOC analyst agents.

You will receive:
1. The agent's documented weaknesses (failure patterns from recent episodes)
2. Available attack types and MITRE techniques
3. A target difficulty level

Design a scenario that:
- EXPLOITS the agent's specific weaknesses
- Has a clear multi-phase kill chain with realistic progression
- Includes 1-3 false positive alerts to test triage accuracy
- Uses real MITRE ATT&CK techniques
- Is challenging but solvable with correct SOC procedure

Rules:
- Only use the provided attack_types and target_nodes
- Red herrings should look similar to real threats but have benign forensic evidence
- Higher difficulty = more phases, more false positives, shorter time window
- The scenario must be uniquely different from common scenarios (no generic "brute force only")

Respond ONLY with valid JSON matching the schema exactly."""

DESIGNER_USER_TEMPLATE = """Agent Weaknesses (from recent failed episodes):
{weaknesses}

Available Attack Types: {attack_types}
Available Target Nodes: {target_nodes}
Target Difficulty: {difficulty}
Max Steps Allowed: {max_steps}

Design a unique attack scenario. Return JSON:
{{
  "scenario_id": "generated_<short_unique_name>",
  "name": "<Descriptive Scenario Name>",
  "description": "<2-3 sentence scenario briefing for the agent>",
  "difficulty": "{difficulty}",
  "threat_count": <int, number of real attack phases>,
  "false_positive_count": <int, 1-4>,
  "adversary_behavior": "<static|evasive|persistent|adaptive>",
  "mitre_techniques_covered": ["T<id>", ...],
  "attack_phases": [
    {{
      "phase_id": "<unique_id>",
      "name": "<Phase Name>",
      "description": "<What the attacker does>",
      "target_node_id": "<node from available nodes>",
      "attack_type": "<type from available types>",
      "steps_to_complete": <int 2-8>,
      "is_active": <true for first phase only>,
      "prerequisite_phase_id": "<prior phase_id or null>",
      "mitre_technique_id": "T<id>",
      "mitre_technique_name": "<technique name>",
      "mitre_tactic": "<tactic name>",
      "c2_ip_pool": ["<ip>"] or [],
      "exfiltration_rate_mb": <float or 0.0>,
      "recompromise_delay": <int or 0>
    }}
  ],
  "initial_compromised_nodes": ["<node_id>"] or []
}}"""


# Available building blocks the designer can choose from
AVAILABLE_ATTACK_TYPES = [
    "intrusion", "malware", "exfiltration", "lateral_movement",
    "privilege_escalation", "brute_force", "phishing", "ransomware", "anomalous_traffic"
]

AVAILABLE_NODES = [
    "web-01", "dc-01", "mail-01", "app-01", "db-01",
    "backup-01", "ws-01", "ws-02", "ws-03", "ws-04"
]

DIFFICULTY_MAX_STEPS = {
    "easy": 15, "medium": 20, "hard": 30, "nightmare": 45
}


@dataclass
class FailureLog:
    """Represents a failed/low-scoring agent episode."""
    scenario_id: str
    score: float
    actions: list[dict]
    failure_reason: str
    common_mistakes: list[str]


class AttackDesigner:
    """
    LLM-powered adversarial attack scenario generator.

    Analyzes agent weaknesses and generates targeted training scenarios.
    Falls back to a deterministic template generator if no LLM is available.
    """

    def __init__(self) -> None:
        self._api_base = os.getenv("API_BASE_URL", "").rstrip("/")
        self._model = os.getenv("MODEL_NAME", "")
        self._token = os.getenv("HF_TOKEN", "") or os.getenv("OPENAI_API_KEY", "")
        self._enabled = bool(self._api_base and self._model and self._token)
        self._rng = random.Random(int(time.time()))

    def design_scenario(
        self,
        failure_logs: list[dict],
        target_difficulty: str = "hard",
    ) -> Optional[ScenarioConfig]:
        """
        Design a new attack scenario targeting agent weaknesses.

        Args:
            failure_logs: List of dicts with keys: scenario_id, score, actions, failure_reason
            target_difficulty: 'easy'|'medium'|'hard'|'nightmare'

        Returns:
            ScenarioConfig ready to register, or None if generation fails
        """
        weaknesses = self._analyze_weaknesses(failure_logs)

        if self._enabled:
            return self._design_with_llm(weaknesses, target_difficulty)
        else:
            return self._design_fallback(weaknesses, target_difficulty)

    def _analyze_weaknesses(self, failure_logs: list[dict]) -> str:
        """Extract weakness patterns from failure logs."""
        if not failure_logs:
            return "- Agent has not yet failed — design a moderately challenging scenario."

        patterns = []
        action_counts: dict[str, int] = {}

        for log in failure_logs:
            score = log.get("score", 0.0)
            reason = log.get("failure_reason", "unknown failure")
            if score < 0.5:
                patterns.append(f"- CRITICAL failure on '{log.get('scenario_id', '?')}' (score: {score:.2f}): {reason}")
            elif score < 0.75:
                patterns.append(f"- Suboptimal on '{log.get('scenario_id', '?')}' (score: {score:.2f}): {reason}")

            # Count action frequencies
            for action in log.get("actions", []):
                act = action.get("action", "unknown")
                action_counts[act] = action_counts.get(act, 0) + 1

        # Identify over/under-used actions
        if action_counts:
            most_used = max(action_counts, key=lambda k: action_counts[k])
            least_used = min(action_counts, key=lambda k: action_counts[k])
            patterns.append(f"- Agent over-relies on '{most_used}' ({action_counts[most_used]}x)")
            patterns.append(f"- Agent under-uses '{least_used}' ({action_counts[least_used]}x)")

        return "\n".join(patterns) if patterns else "- No clear weakness pattern detected."

    def _design_with_llm(self, weaknesses: str, difficulty: str) -> Optional[ScenarioConfig]:
        """Use LLM to design a targeted scenario."""
        max_steps = DIFFICULTY_MAX_STEPS.get(difficulty, 25)

        prompt = DESIGNER_USER_TEMPLATE.format(
            weaknesses=weaknesses,
            attack_types=", ".join(AVAILABLE_ATTACK_TYPES),
            target_nodes=", ".join(AVAILABLE_NODES),
            difficulty=difficulty,
            max_steps=max_steps,
        )

        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=self._api_base + "/v1",
                api_key=self._token,
            )
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": DESIGNER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.7,  # Some randomness for variety
            )
            raw = response.choices[0].message.content or ""
            return self._parse_scenario(raw, max_steps)
        except Exception as e:
            print(f"[AttackDesigner] LLM generation failed: {e} — using fallback")
            return self._design_fallback(weaknesses, difficulty)

    def _parse_scenario(self, raw: str, max_steps: int) -> Optional[ScenarioConfig]:
        """Parse LLM JSON output into ScenarioConfig."""
        # Strip markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start < 0 or end <= start:
            return None

        try:
            data = json.loads(raw[start:end])
        except json.JSONDecodeError:
            return None

        # Ensure unique scenario_id with timestamp
        scenario_id = data.get("scenario_id", f"generated_{int(time.time())}")
        if not scenario_id.startswith("generated_"):
            scenario_id = f"generated_{scenario_id}"

        # Map difficulty string to enum
        diff_map = {
            "easy": Difficulty.EASY, "medium": Difficulty.MEDIUM,
            "hard": Difficulty.HARD, "nightmare": Difficulty.NIGHTMARE,
        }
        difficulty = diff_map.get(data.get("difficulty", "hard"), Difficulty.HARD)

        # Build attack phases
        phases = []
        for i, ph in enumerate(data.get("attack_phases", [])):
            # Map attack_type string to AlertType enum
            type_map = {t: AlertType(t) for t in AVAILABLE_ATTACK_TYPES if t in [a.value for a in AlertType]}
            attack_type = type_map.get(ph.get("attack_type", "intrusion"), AlertType.INTRUSION)

            phases.append(AttackPhase(
                phase_id=ph.get("phase_id", f"gen-{i+1:02d}"),
                name=ph.get("name", f"Phase {i+1}"),
                description=ph.get("description", ""),
                target_node_id=ph.get("target_node_id", "web-01"),
                attack_type=attack_type,
                steps_to_complete=max(2, min(10, int(ph.get("steps_to_complete", 4)))),
                is_active=bool(ph.get("is_active", i == 0)),
                prerequisite_phase_id=ph.get("prerequisite_phase_id") or None,
                mitre_technique_id=ph.get("mitre_technique_id", "T0000"),
                mitre_technique_name=ph.get("mitre_technique_name", "Unknown"),
                mitre_tactic=ph.get("mitre_tactic", "unknown"),
                c2_ip_pool=ph.get("c2_ip_pool", []),
                exfiltration_rate_mb=float(ph.get("exfiltration_rate_mb", 0.0)),
                recompromise_delay=int(ph.get("recompromise_delay", 0)),
            ))

        if not phases:
            return None

        return ScenarioConfig(
            scenario_id=scenario_id,
            name=data.get("name", "Generated Scenario"),
            description=data.get("description", "A dynamically generated attack scenario."),
            difficulty=difficulty,
            threat_count=data.get("threat_count", len(phases)),
            false_positive_count=max(0, min(4, int(data.get("false_positive_count", 1)))),
            max_steps=max_steps,
            adversary_behavior=data.get("adversary_behavior", "evasive"),
            mitre_techniques_covered=data.get("mitre_techniques_covered", []),
            attack_phases=phases,
            initial_compromised_nodes=data.get("initial_compromised_nodes", []),
        )

    def _design_fallback(self, weaknesses: str, difficulty: str) -> ScenarioConfig:
        """
        Deterministic fallback: rotate through handcrafted weakness-targeting templates.
        Used when no LLM API is configured.
        """
        max_steps = DIFFICULTY_MAX_STEPS.get(difficulty, 25)
        diff_enum = {
            "easy": Difficulty.EASY, "medium": Difficulty.MEDIUM,
            "hard": Difficulty.HARD, "nightmare": Difficulty.NIGHTMARE,
        }.get(difficulty, Difficulty.HARD)

        timestamp = int(time.time())
        # Pick a template based on detected weakness keywords
        if "FP" in weaknesses or "false positive" in weaknesses.lower() or "dismiss" in weaknesses.lower():
            return self._template_fp_heavy(timestamp, diff_enum, max_steps)
        elif "exfil" in weaknesses.lower() or "data" in weaknesses.lower():
            return self._template_fast_exfil(timestamp, diff_enum, max_steps)
        else:
            return self._template_multi_vector(timestamp, diff_enum, max_steps)

    def _template_fp_heavy(self, ts: int, difficulty: Difficulty, max_steps: int) -> ScenarioConfig:
        """Template: high FP count to punish agents that block before investigating."""
        return ScenarioConfig(
            scenario_id=f"generated_fp_gauntlet_{ts}",
            name="False Positive Gauntlet",
            description=(
                "A low-and-slow APT is hiding within a flood of routine security alerts. "
                "Three out of four alerts are false positives — automated scanner noise, "
                "legitimate pentest activity, and routine maintenance. Only one alert is real: "
                "a credential theft on the domain controller. Can you find it without false moves?"
            ),
            difficulty=difficulty,
            threat_count=1,
            false_positive_count=3,
            max_steps=max_steps,
            adversary_behavior="evasive",
            mitre_techniques_covered=["T1078.002", "T1003.001"],
            attack_phases=[
                AttackPhase(
                    phase_id="fpg-01",
                    name="Credential Theft on DC",
                    description="APT actor using valid domain credentials to dump LSASS on dc-01",
                    target_node_id="dc-01",
                    attack_type=AlertType.PRIVILEGE_ESCALATION,
                    steps_to_complete=6,
                    is_active=True,
                    mitre_technique_id="T1003.001",
                    mitre_technique_name="OS Credential Dumping: LSASS Memory",
                    mitre_tactic="credential-access",
                    c2_ip_pool=["203.0.113.55"],
                ),
            ],
            initial_compromised_nodes=[],
        )

    def _template_fast_exfil(self, ts: int, difficulty: Difficulty, max_steps: int) -> ScenarioConfig:
        """Template: rapid exfiltration to punish slow agents."""
        return ScenarioConfig(
            scenario_id=f"generated_speed_exfil_{ts}",
            name="High-Speed Database Exfiltration",
            description=(
                "An attacker with pre-positioned access has begun mass-exfiltrating the "
                "customer database at 15 MB/step. You have a narrow window to contain the breach "
                "before irreversible data loss occurs. The attacker has already bypassed "
                "the perimeter — internal lateral movement is ongoing."
            ),
            difficulty=difficulty,
            threat_count=2,
            false_positive_count=1,
            max_steps=max_steps,
            adversary_behavior="persistent",
            mitre_techniques_covered=["T1041", "T1021.002"],
            attack_phases=[
                AttackPhase(
                    phase_id="exf-01",
                    name="Lateral Movement to Database",
                    description="Attacker moving laterally from app-01 to db-01",
                    target_node_id="db-01",
                    attack_type=AlertType.LATERAL_MOVEMENT,
                    steps_to_complete=3,
                    is_active=True,
                    mitre_technique_id="T1021.002",
                    mitre_technique_name="Remote Services: SMB Admin Shares",
                    mitre_tactic="lateral-movement",
                ),
                AttackPhase(
                    phase_id="exf-02",
                    name="Mass Database Exfiltration",
                    description="Streaming customer records to external C2 at 15 MB/step",
                    target_node_id="db-01",
                    attack_type=AlertType.EXFILTRATION,
                    steps_to_complete=8,
                    is_active=False,
                    prerequisite_phase_id="exf-01",
                    mitre_technique_id="T1041",
                    mitre_technique_name="Exfiltration Over C2 Channel",
                    mitre_tactic="exfiltration",
                    exfiltration_rate_mb=15.0,
                    c2_ip_pool=["91.219.236.200"],
                ),
            ],
            initial_compromised_nodes=["app-01"],
        )

    def _template_multi_vector(self, ts: int, difficulty: Difficulty, max_steps: int) -> ScenarioConfig:
        """Template: simultaneous multi-vector attack."""
        return ScenarioConfig(
            scenario_id=f"generated_multi_vector_{ts}",
            name="Simultaneous Multi-Vector Intrusion",
            description=(
                "Two independent threat actors are operating simultaneously on your network. "
                "A ransomware gang is targeting workstations via phishing while an APT group "
                "exploits the web server. Prioritize correctly — one threat will detonate before "
                "you can contain both if resources are misallocated."
            ),
            difficulty=difficulty,
            threat_count=3,
            false_positive_count=2,
            max_steps=max_steps,
            adversary_behavior="adaptive",
            mitre_techniques_covered=["T1486", "T1190", "T1566.001"],
            attack_phases=[
                AttackPhase(
                    phase_id="mv-01",
                    name="Web Server Exploit",
                    description="APT exploiting CVE in web-01 for initial foothold",
                    target_node_id="web-01",
                    attack_type=AlertType.INTRUSION,
                    steps_to_complete=3,
                    is_active=True,
                    mitre_technique_id="T1190",
                    mitre_technique_name="Exploit Public-Facing Application",
                    mitre_tactic="initial-access",
                    c2_ip_pool=["198.51.100.77"],
                ),
                AttackPhase(
                    phase_id="mv-02",
                    name="Phishing-Delivered Ransomware",
                    description="Ransomware payload delivered via phishing to ws-03",
                    target_node_id="ws-03",
                    attack_type=AlertType.RANSOMWARE,
                    steps_to_complete=4,
                    is_active=True,
                    mitre_technique_id="T1566.001",
                    mitre_technique_name="Phishing: Spearphishing Attachment",
                    mitre_tactic="initial-access",
                ),
                AttackPhase(
                    phase_id="mv-03",
                    name="Ransomware Lateral Spread",
                    description="Ransomware spreading from ws-03 to ws-04",
                    target_node_id="ws-04",
                    attack_type=AlertType.RANSOMWARE,
                    steps_to_complete=3,
                    is_active=False,
                    prerequisite_phase_id="mv-02",
                    mitre_technique_id="T1486",
                    mitre_technique_name="Data Encrypted for Impact",
                    mitre_tactic="impact",
                ),
            ],
            initial_compromised_nodes=["ws-03"],
        )


# ─────────────────────────────────────────────────────────────
# Curriculum Manager — tracks progress and decides when to
# unlock harder / generated scenarios
# ─────────────────────────────────────────────────────────────

class CurriculumManager:
    """
    Manages progressive difficulty for the attack designer.

    Tracks rolling average scores per scenario type and flags
    when the agent is ready for harder generated challenges.
    """

    THRESHOLDS = {
        "easy": 0.85,
        "medium": 0.80,
        "hard": 0.75,
        "nightmare": 0.70,
    }

    def __init__(self) -> None:
        self._history: dict[str, list[float]] = {}
        self._window = 5  # rolling window size

    def record_score(self, scenario_id: str, score: float) -> None:
        """Record a score for a scenario."""
        if scenario_id not in self._history:
            self._history[scenario_id] = []
        self._history[scenario_id].append(score)
        # Keep only last N
        self._history[scenario_id] = self._history[scenario_id][-self._window:]

    def get_rolling_average(self, scenario_id: str) -> float:
        """Get the rolling average score for a scenario."""
        history = self._history.get(scenario_id, [])
        return sum(history) / len(history) if history else 0.0

    def get_weakness_scenarios(self) -> list[str]:
        """Return scenario IDs where the agent is still struggling."""
        weak = []
        for sid, scores in self._history.items():
            avg = sum(scores) / len(scores) if scores else 0.0
            if avg < 0.70:
                weak.append(sid)
        return weak

    def should_generate_new_scenario(self, difficulty: str) -> bool:
        """Return True if agent is ready for a generated challenge at this difficulty."""
        threshold = self.THRESHOLDS.get(difficulty, 0.80)
        relevant = [
            sid for sid in self._history
            if difficulty in sid or len(self._history[sid]) >= 3
        ]
        if not relevant:
            return False
        averages = [self.get_rolling_average(sid) for sid in relevant]
        return sum(averages) / len(averages) >= threshold

    def get_failure_logs(self) -> list[dict]:
        """Build failure log dicts for the AttackDesigner."""
        logs = []
        for sid, scores in self._history.items():
            if scores and scores[-1] < 0.75:
                logs.append({
                    "scenario_id": sid,
                    "score": scores[-1],
                    "actions": [],
                    "failure_reason": f"Rolling avg: {self.get_rolling_average(sid):.2f}",
                })
        return logs
