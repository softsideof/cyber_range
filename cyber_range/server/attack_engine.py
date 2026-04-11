"""Attack scenarios, progression engine, and deterministic grading."""


import random
from typing import Optional

try:
    from ..models import (
        AlertSeverity, AlertType, AttackPhase, Difficulty,
        EpisodeMetrics, NetworkAlert, NodeStatus, ScenarioConfig,
    )
    from .network_simulator import NetworkSimulator
except ImportError:
    from cyber_range.models import (
        AlertSeverity, AlertType, AttackPhase, Difficulty,
        EpisodeMetrics, NetworkAlert, NodeStatus, ScenarioConfig,
    )
    from cyber_range.server.network_simulator import NetworkSimulator


# Scenario definitions

SCENARIOS: dict[str, ScenarioConfig] = {
    "script_kiddie": ScenarioConfig(
        scenario_id="script_kiddie",
        name="Script Kiddie Brute Force",
        description=(
            "A script kiddie is running an SSH brute force attack against your web server "
            "(web-01) from external IP 185.220.101.42. Three clear alerts have fired. "
            "Investigate the alerts, block the attacker IP, and ensure the web server is secure."
        ),
        difficulty=Difficulty.EASY,
        threat_count=1,
        false_positive_count=1,
        max_steps=15,
        adversary_behavior="static",
        mitre_techniques_covered=["T1110.001"],
        attack_phases=[
            AttackPhase(
                phase_id="bf-01",
                name="SSH Brute Force",
                description="Automated SSH login attempts against web-01",
                target_node_id="web-01",
                attack_type=AlertType.BRUTE_FORCE,
                steps_to_complete=8,
                is_active=True,
                mitre_technique_id="T1110.001",
                mitre_technique_name="Brute Force: Password Guessing",
                mitre_tactic="credential-access",
            ),
        ],
        initial_compromised_nodes=[],
    ),

    "phishing_campaign": ScenarioConfig(
        scenario_id="phishing_campaign",
        name="Phishing Campaign Triage",
        description=(
            "A targeted phishing campaign has hit your organization. Five alerts have been "
            "generated — 3 are from actual malware infections on workstations that clicked "
            "the phishing link, and 2 are false positives from legitimate email activity. "
            "Investigate each alert, quarantine truly compromised hosts, and correctly "
            "dismiss false positives. Watch out — malware may try to spread laterally."
        ),
        difficulty=Difficulty.MEDIUM,
        threat_count=3,
        false_positive_count=2,
        max_steps=25,
        adversary_behavior="evasive",
        mitre_techniques_covered=["T1566.001", "T1204.002", "T1021.002"],
        attack_phases=[
            AttackPhase(
                phase_id="phish-01",
                name="Malware Infection - ws-01",
                description="Workstation ws-01 clicked phishing link, malware installed",
                target_node_id="ws-01",
                attack_type=AlertType.PHISHING,
                steps_to_complete=6,
                is_active=True,
                mitre_technique_id="T1566.001",
                mitre_technique_name="Phishing: Spearphishing Attachment",
                mitre_tactic="initial-access",
            ),
            AttackPhase(
                phase_id="phish-02",
                name="Malware Infection - ws-02",
                description="Workstation ws-02 clicked phishing link, malware installed",
                target_node_id="ws-02",
                attack_type=AlertType.PHISHING,
                steps_to_complete=6,
                is_active=True,
                mitre_technique_id="T1204.002",
                mitre_technique_name="User Execution: Malicious File",
                mitre_tactic="execution",
            ),
            AttackPhase(
                phase_id="phish-03",
                name="Lateral Movement Attempt",
                description="Malware attempts to spread from ws-01 to app-01 via SMB",
                target_node_id="app-01",
                attack_type=AlertType.LATERAL_MOVEMENT,
                steps_to_complete=5,
                is_active=False,
                prerequisite_phase_id="phish-01",
                mitre_technique_id="T1021.002",
                mitre_technique_name="Remote Services: SMB/Windows Admin Shares",
                mitre_tactic="lateral-movement",
            ),
        ],
        initial_compromised_nodes=["ws-01", "ws-02"],
    ),

    "apt_lateral_movement": ScenarioConfig(
        scenario_id="apt_lateral_movement",
        name="APT Kill Chain — Lateral Movement",
        description=(
            "An Advanced Persistent Threat (APT) group has gained initial access through a "
            "compromised web server (web-01). They are executing a multi-stage attack: "
            "credential harvesting → lateral movement to domain controller → privilege "
            "escalation → data exfiltration from the production database. "
            "You must trace the full kill chain, contain the threat at each stage, "
            "and prevent data exfiltration. The adversary uses C2 IP rotation to evade blocking."
        ),
        difficulty=Difficulty.HARD,
        threat_count=5,
        false_positive_count=3,
        max_steps=35,
        adversary_behavior="evasive",
        mitre_techniques_covered=["T1190", "T1003.001", "T1021.002", "T1078.002", "T1041"],
        attack_phases=[
            AttackPhase(
                phase_id="apt-01",
                name="Initial Access via Web Server",
                description="APT exploited CVE-2024-1234 on web-01, reverse shell established",
                target_node_id="web-01",
                attack_type=AlertType.INTRUSION,
                steps_to_complete=3,
                is_active=True,
                mitre_technique_id="T1190",
                mitre_technique_name="Exploit Public-Facing Application",
                mitre_tactic="initial-access",
                c2_ip_pool=["91.219.236.166", "198.51.100.23"],
            ),
            AttackPhase(
                phase_id="apt-02",
                name="Credential Harvesting",
                description="Dumping credentials from web-01 via Mimikatz (LSASS memory)",
                target_node_id="web-01",
                attack_type=AlertType.PRIVILEGE_ESCALATION,
                steps_to_complete=4,
                is_active=False,
                prerequisite_phase_id="apt-01",
                mitre_technique_id="T1003.001",
                mitre_technique_name="OS Credential Dumping: LSASS Memory",
                mitre_tactic="credential-access",
                recompromise_delay=3,
            ),
            AttackPhase(
                phase_id="apt-03",
                name="Lateral Movement to DC",
                description="Using stolen credentials to move to domain controller dc-01 via SMB",
                target_node_id="dc-01",
                attack_type=AlertType.LATERAL_MOVEMENT,
                steps_to_complete=5,
                is_active=False,
                prerequisite_phase_id="apt-02",
                mitre_technique_id="T1021.002",
                mitre_technique_name="Remote Services: SMB/Windows Admin Shares",
                mitre_tactic="lateral-movement",
            ),
            AttackPhase(
                phase_id="apt-04",
                name="Privilege Escalation on DC",
                description="Elevating to Domain Admin on dc-01 via token impersonation",
                target_node_id="dc-01",
                attack_type=AlertType.PRIVILEGE_ESCALATION,
                steps_to_complete=3,
                is_active=False,
                prerequisite_phase_id="apt-03",
                mitre_technique_id="T1078.002",
                mitre_technique_name="Valid Accounts: Domain Accounts",
                mitre_tactic="privilege-escalation",
            ),
            AttackPhase(
                phase_id="apt-05",
                name="Data Exfiltration",
                description="Exfiltrating sensitive data from db-01 to external C2 over HTTPS",
                target_node_id="db-01",
                attack_type=AlertType.EXFILTRATION,
                steps_to_complete=6,
                is_active=False,
                exfiltration_rate_mb=5.0,
                prerequisite_phase_id="apt-04",
                mitre_technique_id="T1041",
                mitre_technique_name="Exfiltration Over C2 Channel",
                mitre_tactic="exfiltration",
                c2_ip_pool=["203.0.113.45", "198.51.100.99"],
            ),
        ],
        initial_compromised_nodes=["web-01"],
    ),

    "ransomware_outbreak": ScenarioConfig(
        scenario_id="ransomware_outbreak",
        name="Ransomware Outbreak",
        description=(
            "A ransomware payload has detonated on workstation ws-01 after a drive-by "
            "download. The malware is spreading rapidly across the network via SMB, "
            "encrypting hosts every 2 steps. You face extreme time pressure: isolate "
            "aggressively (causing business disruption) or attempt targeted containment "
            "(risking further spread). The backup server (backup-01) is the last line "
            "of defense — if it gets encrypted, recovery becomes impossible."
        ),
        difficulty=Difficulty.HARD,
        threat_count=4,
        false_positive_count=1,
        max_steps=20,
        adversary_behavior="persistent",
        mitre_techniques_covered=["T1486", "T1021.002", "T1490", "T1489"],
        attack_phases=[
            AttackPhase(
                phase_id="ransom-01",
                name="Initial Detonation on ws-01",
                description="Ransomware payload executed on ws-01, files being encrypted",
                target_node_id="ws-01",
                attack_type=AlertType.RANSOMWARE,
                steps_to_complete=2,
                is_active=True,
                mitre_technique_id="T1486",
                mitre_technique_name="Data Encrypted for Impact",
                mitre_tactic="impact",
            ),
            AttackPhase(
                phase_id="ransom-02",
                name="Lateral Spread to ws-02",
                description="Ransomware spreading via SMB to ws-02",
                target_node_id="ws-02",
                attack_type=AlertType.RANSOMWARE,
                steps_to_complete=3,
                is_active=False,
                prerequisite_phase_id="ransom-01",
                mitre_technique_id="T1021.002",
                mitre_technique_name="Remote Services: SMB/Windows Admin Shares",
                mitre_tactic="lateral-movement",
            ),
            AttackPhase(
                phase_id="ransom-03",
                name="Lateral Spread to app-01",
                description="Ransomware attempting to encrypt app-01 via network share",
                target_node_id="app-01",
                attack_type=AlertType.RANSOMWARE,
                steps_to_complete=3,
                is_active=False,
                prerequisite_phase_id="ransom-02",
                mitre_technique_id="T1490",
                mitre_technique_name="Inhibit System Recovery",
                mitre_tactic="impact",
            ),
            AttackPhase(
                phase_id="ransom-04",
                name="Targeting Backup Server",
                description="Ransomware attempting to encrypt backup-01 to prevent recovery",
                target_node_id="backup-01",
                attack_type=AlertType.RANSOMWARE,
                steps_to_complete=4,
                is_active=False,
                prerequisite_phase_id="ransom-03",
                mitre_technique_id="T1489",
                mitre_technique_name="Service Stop",
                mitre_tactic="impact",
            ),
        ],
        initial_compromised_nodes=["ws-01"],
    ),

    "insider_threat_apt": ScenarioConfig(
        scenario_id="insider_threat_apt",
        name="Insider Threat + External APT (Dual Threat)",
        description=(
            "Two simultaneous threats are targeting your network. A malicious insider "
            "on the executive workstation (ws-04) is slowly exfiltrating sensitive HR "
            "and financial data to a personal cloud storage account. Meanwhile, an "
            "external APT group has compromised the mail server (mail-01) through a "
            "spear-phishing attack and is moving laterally toward the domain controller. "
            "The adversary adapts: blocking IPs triggers C2 rotation, and patching without "
            "full restoration allows re-compromise. Four false positive alerts add noise."
        ),
        difficulty=Difficulty.NIGHTMARE,
        threat_count=7,
        false_positive_count=4,
        max_steps=45,
        adversary_behavior="adaptive",
        mitre_techniques_covered=[
            "T1567.002", "T1074.001", "T1566.001", "T1003.001",
            "T1021.002", "T1078.002", "T1041",
        ],
        attack_phases=[
            # === Insider Threat Chain ===
            AttackPhase(
                phase_id="insider-01",
                name="Insider Data Staging",
                description="Insider on ws-04 copying sensitive files to staging directory",
                target_node_id="ws-04",
                attack_type=AlertType.ANOMALOUS_TRAFFIC,
                steps_to_complete=5,
                is_active=True,
                mitre_technique_id="T1074.001",
                mitre_technique_name="Data Staged: Local Data Staging",
                mitre_tactic="collection",
            ),
            AttackPhase(
                phase_id="insider-02",
                name="Insider Data Exfiltration",
                description="Insider exfiltrating staged data to personal cloud storage",
                target_node_id="ws-04",
                attack_type=AlertType.EXFILTRATION,
                steps_to_complete=8,
                exfiltration_rate_mb=3.0,
                is_active=False,
                prerequisite_phase_id="insider-01",
                mitre_technique_id="T1567.002",
                mitre_technique_name="Exfiltration to Cloud Storage",
                mitre_tactic="exfiltration",
            ),
            # === External APT Chain ===
            AttackPhase(
                phase_id="ext-apt-01",
                name="Mail Server Compromise",
                description="APT gained access to mail-01 via spear-phishing attachment",
                target_node_id="mail-01",
                attack_type=AlertType.PHISHING,
                steps_to_complete=3,
                is_active=True,
                mitre_technique_id="T1566.001",
                mitre_technique_name="Phishing: Spearphishing Attachment",
                mitre_tactic="initial-access",
                c2_ip_pool=["91.219.236.166", "198.51.100.23", "203.0.113.77"],
            ),
            AttackPhase(
                phase_id="ext-apt-02",
                name="Credential Harvesting from Mail",
                description="Extracting cached credentials from mail-01 via LSASS dump",
                target_node_id="mail-01",
                attack_type=AlertType.PRIVILEGE_ESCALATION,
                steps_to_complete=4,
                is_active=False,
                prerequisite_phase_id="ext-apt-01",
                mitre_technique_id="T1003.001",
                mitre_technique_name="OS Credential Dumping: LSASS Memory",
                mitre_tactic="credential-access",
                recompromise_delay=3,
            ),
            AttackPhase(
                phase_id="ext-apt-03",
                name="Lateral Movement to DC",
                description="Using stolen creds to access domain controller dc-01 via SMB",
                target_node_id="dc-01",
                attack_type=AlertType.LATERAL_MOVEMENT,
                steps_to_complete=5,
                is_active=False,
                prerequisite_phase_id="ext-apt-02",
                mitre_technique_id="T1021.002",
                mitre_technique_name="Remote Services: SMB/Windows Admin Shares",
                mitre_tactic="lateral-movement",
            ),
            AttackPhase(
                phase_id="ext-apt-04",
                name="Domain Admin Escalation",
                description="Elevating privileges to Domain Admin on dc-01",
                target_node_id="dc-01",
                attack_type=AlertType.PRIVILEGE_ESCALATION,
                steps_to_complete=3,
                is_active=False,
                prerequisite_phase_id="ext-apt-03",
                mitre_technique_id="T1078.002",
                mitre_technique_name="Valid Accounts: Domain Accounts",
                mitre_tactic="privilege-escalation",
            ),
            AttackPhase(
                phase_id="ext-apt-05",
                name="Mass Data Exfiltration",
                description="Exfiltrating entire database via domain admin access over C2",
                target_node_id="db-01",
                attack_type=AlertType.EXFILTRATION,
                steps_to_complete=6,
                exfiltration_rate_mb=10.0,
                is_active=False,
                prerequisite_phase_id="ext-apt-04",
                mitre_technique_id="T1041",
                mitre_technique_name="Exfiltration Over C2 Channel",
                mitre_tactic="exfiltration",
                c2_ip_pool=["203.0.113.45", "198.51.100.99"],
            ),
        ],
        initial_compromised_nodes=["mail-01"],
    ),

    "supply_chain_compromise": ScenarioConfig(
        scenario_id="supply_chain_compromise",
        name="Supply Chain Compromise",
        description=(
            "A trusted software update for the application server (app-01) has been "
            "trojaned by an advanced threat actor. The compromised update installed a "
            "backdoor that is beaconing to C2 infrastructure. The attacker is using "
            "PowerShell for post-exploitation, downloading additional tools, and "
            "pivoting toward the production database (db-01) to exfiltrate customer "
            "records. The initial alert has HIGH confidence but the source appears to "
            "be a legitimate update process, making triage critical. Two false "
            "positives from routine update checks add noise."
        ),
        difficulty=Difficulty.HARD,
        threat_count=4,
        false_positive_count=2,
        max_steps=30,
        adversary_behavior="evasive",
        mitre_techniques_covered=[
            "T1195.002", "T1059.001", "T1105", "T1041",
        ],
        attack_phases=[
            AttackPhase(
                phase_id="supply-01",
                name="Trojaned Update Execution",
                description="Backdoored software update executed on app-01 via trusted channel",
                target_node_id="app-01",
                attack_type=AlertType.MALWARE,
                steps_to_complete=3,
                is_active=True,
                mitre_technique_id="T1195.002",
                mitre_technique_name="Supply Chain Compromise: Compromise Software Supply Chain",
                mitre_tactic="initial-access",
                c2_ip_pool=["198.51.100.88", "203.0.113.99"],
            ),
            AttackPhase(
                phase_id="supply-02",
                name="PowerShell Post-Exploitation",
                description="Backdoor using PowerShell to enumerate internal network from app-01",
                target_node_id="app-01",
                attack_type=AlertType.PRIVILEGE_ESCALATION,
                steps_to_complete=4,
                is_active=False,
                prerequisite_phase_id="supply-01",
                mitre_technique_id="T1059.001",
                mitre_technique_name="Command and Scripting Interpreter: PowerShell",
                mitre_tactic="execution",
                recompromise_delay=3,
            ),
            AttackPhase(
                phase_id="supply-03",
                name="Ingress Tool Transfer",
                description="Downloading additional tools (Cobalt Strike beacon) to app-01",
                target_node_id="app-01",
                attack_type=AlertType.INTRUSION,
                steps_to_complete=4,
                is_active=False,
                prerequisite_phase_id="supply-02",
                mitre_technique_id="T1105",
                mitre_technique_name="Ingress Tool Transfer",
                mitre_tactic="command-and-control",
            ),
            AttackPhase(
                phase_id="supply-04",
                name="Database Exfiltration",
                description="Pivoting from app-01 to db-01, exfiltrating customer records via HTTPS",
                target_node_id="db-01",
                attack_type=AlertType.EXFILTRATION,
                steps_to_complete=6,
                exfiltration_rate_mb=8.0,
                is_active=False,
                prerequisite_phase_id="supply-03",
                mitre_technique_id="T1041",
                mitre_technique_name="Exfiltration Over C2 Channel",
                mitre_tactic="exfiltration",
                c2_ip_pool=["198.51.100.88", "203.0.113.99"],
            ),
        ],
        initial_compromised_nodes=["app-01"],
    ),
}



class AttackEngine:
    """
    Manages attack scenario lifecycle, progression, and grading.

    The attacker progresses each step — if the agent doesn't act fast enough,
    the attacker advances through their kill chain.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self.scenario: Optional[ScenarioConfig] = None
        self.phases: list[AttackPhase] = []
        self.metrics = EpisodeMetrics()
        self._total_exfiltrated_mb: float = 0.0
        self._total_prevented_mb: float = 0.0
        self._attacker_ips: set[str] = set()

    def get_available_scenarios(self) -> list[dict]:
        """Return list of available scenario IDs and descriptions."""
        return [
            {
                "scenario_id": s.scenario_id,
                "name": s.name,
                "difficulty": s.difficulty.value,
                "description": s.description,
                "max_steps": s.max_steps,
            }
            for s in SCENARIOS.values()
        ]

    def load_scenario(
        self, scenario_id: str, network: NetworkSimulator, seed: Optional[int] = None
    ) -> ScenarioConfig:
        """
        Load and initialize a specific attack scenario.

        Args:
            scenario_id: ID of the scenario to load.
            network: NetworkSimulator to apply initial compromises to.
            seed: Optional random seed.

        Returns:
            The loaded ScenarioConfig.

        Raises:
            ValueError: If scenario_id not found.
        """
        if scenario_id not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario: {scenario_id}. "
                f"Available: {list(SCENARIOS.keys())}"
            )

        if seed is not None:
            self._rng = random.Random(seed)

        self.scenario = SCENARIOS[scenario_id]
        # Deep-copy phases so we can mutate state
        self.phases = [
            AttackPhase(
                phase_id=p.phase_id, name=p.name, description=p.description,
                target_node_id=p.target_node_id, attack_type=p.attack_type,
                steps_to_complete=p.steps_to_complete,
                exfiltration_rate_mb=p.exfiltration_rate_mb,
                is_active=p.is_active, prerequisite_phase_id=p.prerequisite_phase_id,
                mitre_technique_id=p.mitre_technique_id,
                mitre_technique_name=p.mitre_technique_name,
                mitre_tactic=p.mitre_tactic,
                c2_ip_pool=list(p.c2_ip_pool),
                recompromise_delay=p.recompromise_delay,
            )
            for p in self.scenario.attack_phases
        ]

        # Reset metrics
        self.metrics = EpisodeMetrics(
            total_threats=self.scenario.threat_count,
            false_positives_total=self.scenario.false_positive_count,
            total_attack_chains=len(self.phases),
        )

        self._total_exfiltrated_mb = 0.0
        self._total_prevented_mb = 0.0
        self._adversary_behavior = self.scenario.adversary_behavior

        # Set attacker IPs (base pool + C2 rotation IPs from phases)
        self._attacker_ips = {"185.220.101.42", "94.232.46.19", "45.155.205.233"}
        self._c2_rotation_pool: list[str] = []
        for p in self.phases:
            self._c2_rotation_pool.extend(p.c2_ip_pool)
        self._active_c2_ip: Optional[str] = None  # Currently active rotated C2

        # Apply initial compromises
        for node_id in self.scenario.initial_compromised_nodes:
            network.compromise_node(node_id, step=0)

        # Generate initial alerts
        self._generate_initial_alerts(network)

        return self.scenario

    def _generate_initial_alerts(self, network: NetworkSimulator) -> None:
        """Generate the initial set of SIEM alerts for the loaded scenario."""
        if not self.scenario:
            return

        attacker_ip = self._rng.choice(list(self._attacker_ips))

        # Generate alerts for active attack phases
        for phase in self.phases:
            if not phase.is_active:
                continue

            target_node = network.get_node(phase.target_node_id)
            if not target_node:
                continue

            alert_id = network.generate_alert_id()
            severity = {
                AlertType.BRUTE_FORCE: AlertSeverity.HIGH,
                AlertType.PHISHING: AlertSeverity.MEDIUM,
                AlertType.INTRUSION: AlertSeverity.CRITICAL,
                AlertType.LATERAL_MOVEMENT: AlertSeverity.HIGH,
                AlertType.PRIVILEGE_ESCALATION: AlertSeverity.CRITICAL,
                AlertType.EXFILTRATION: AlertSeverity.CRITICAL,
                AlertType.RANSOMWARE: AlertSeverity.CRITICAL,
                AlertType.ANOMALOUS_TRAFFIC: AlertSeverity.MEDIUM,
            }.get(phase.attack_type, AlertSeverity.MEDIUM)

            raw_logs = {
                AlertType.BRUTE_FORCE: f"sshd[12345]: Failed password for root from {attacker_ip} port 43210 ssh2 (attempt 847/1000)",
                AlertType.PHISHING: f"mail-filter: Suspicious attachment detected from external sender to {target_node.hostname}, payload matched known malware signature Emotet.Gen4",
                AlertType.INTRUSION: f"ids[nginx]: Exploit attempt detected — CVE-2024-1234 reverse shell payload from {attacker_ip} to {target_node.ip_address}:443",
                AlertType.LATERAL_MOVEMENT: f"event_log: Anomalous SMB connection from {target_node.ip_address} to 10.0.1.1 using service account svc_backup",
                AlertType.PRIVILEGE_ESCALATION: f"security_log: Process mimikatz.exe spawned by {target_node.hostname}\\SYSTEM — LSASS memory dump detected",
                AlertType.EXFILTRATION: f"netflow: Unusual outbound data transfer from {target_node.ip_address} to {attacker_ip}:443 — 50MB in 2 minutes",
                AlertType.RANSOMWARE: f"edr[crowdstrike]: CRITICAL — Ransomware behavior detected on {target_node.hostname} — rapid file encryption (.locked extension) across SMB shares",
                AlertType.ANOMALOUS_TRAFFIC: f"dlp[forcepoint]: Unusual data access pattern from {target_node.hostname} — bulk file copy to staging directory detected",
            }

            network.add_alert(NetworkAlert(
                alert_id=alert_id,
                timestamp=0.0,
                severity=severity,
                source_ip=attacker_ip if phase.attack_type in (AlertType.BRUTE_FORCE, AlertType.INTRUSION) else target_node.ip_address,
                destination_ip=target_node.ip_address if phase.attack_type in (AlertType.BRUTE_FORCE, AlertType.INTRUSION) else attacker_ip,
                alert_type=phase.attack_type,
                description=f"[{severity.value.upper()}] {phase.name}: {phase.description}",
                confidence=self._rng.uniform(0.75, 0.98),
                raw_log=raw_logs.get(phase.attack_type, "Unknown log event"),
                is_false_positive=False,
                related_node_id=phase.target_node_id,
                related_attack_phase=phase.phase_id,
            ))

        # Generate false positive alerts
        fp_types = [
            ("Legitimate bulk email delivery", AlertType.ANOMALOUS_TRAFFIC, AlertSeverity.LOW, "mail-01"),
            ("Scheduled backup transfer spike", AlertType.ANOMALOUS_TRAFFIC, AlertSeverity.INFO, "backup-01"),
            ("Developer testing API load", AlertType.ANOMALOUS_TRAFFIC, AlertSeverity.LOW, "app-01"),
        ]

        for i in range(min(self.scenario.false_positive_count, len(fp_types))):
            desc, atype, sev, node_id = fp_types[i]
            target_node = network.get_node(node_id)
            if not target_node:
                continue

            alert_id = network.generate_alert_id()
            network.add_alert(NetworkAlert(
                alert_id=alert_id,
                timestamp=0.0,
                severity=sev,
                source_ip=target_node.ip_address,
                destination_ip="10.0.0.1",
                alert_type=atype,
                description=f"[{sev.value.upper()}] {desc}",
                confidence=self._rng.uniform(0.30, 0.60),
                raw_log=f"netflow: Traffic pattern anomaly from {target_node.ip_address} — {desc.lower()}",
                is_false_positive=True,
                related_node_id=node_id,
            ))

    def advance(self, network: NetworkSimulator) -> list[str]:
        """
        Advance all active attack phases by one step.

        The attacker progresses unless the agent has contained them.
        Also checks completed-but-uncontained phases for post-compromise
        containment (e.g., isolating an already-compromised host still
        counts as threat neutralization).

        Returns a list of event descriptions for logging.
        """
        events: list[str] = []
        if not self.scenario:
            return events

        for phase in self.phases:
            # --- Post-compromise containment check ---
            # If phase already completed but defender subsequently contained it
            # (e.g., isolated the compromised node), credit the neutralization.
            if phase.is_completed and not phase.is_contained:
                target_node = network.get_node(phase.target_node_id)
                if target_node and target_node.status == NodeStatus.ISOLATED:
                    phase.is_contained = True
                    self.metrics.threats_neutralized += 1
                    events.append(
                        f"Attack phase '{phase.name}' neutralized post-compromise "
                        f"— target {phase.target_node_id} isolated."
                    )
                continue  # Completed phases don't advance further

            if not phase.is_active or phase.is_contained:
                continue

            target_node = network.get_node(phase.target_node_id)

            # If target is isolated or the attacker IP is blocked, phase is contained
            if target_node and target_node.status == NodeStatus.ISOLATED:
                phase.is_contained = True
                self.metrics.threats_neutralized += 1
                events.append(f"Attack phase '{phase.name}' contained — target {phase.target_node_id} is isolated.")
                continue

            # Check if attacker IP is blocked for external attacks
            if phase.attack_type in (AlertType.BRUTE_FORCE, AlertType.INTRUSION):
                blocked = any(ip in network.blocked_ips for ip in self._attacker_ips)
                # Adaptive adversary: rotate to backup C2 IP if primary is blocked
                if blocked and self._adversary_behavior in ("evasive", "adaptive"):
                    if self._c2_rotation_pool:
                        new_ip = self._c2_rotation_pool.pop(0)
                        self._attacker_ips.add(new_ip)
                        self._active_c2_ip = new_ip
                        events.append(
                            f"🔄 Adversary rotated C2 to {new_ip} after IP block detected "
                            f"[MITRE: T1571 Non-Standard Port / T1090 Proxy]"
                        )
                        # Generate a new alert for the rotated C2
                        if target_node:
                            alert_id = network.generate_alert_id()
                            network.add_alert(NetworkAlert(
                                alert_id=alert_id,
                                timestamp=float(network.elapsed_steps()),
                                severity=AlertSeverity.HIGH,
                                source_ip=new_ip,
                                destination_ip=target_node.ip_address,
                                alert_type=phase.attack_type,
                                description=f"[HIGH] New C2 channel detected from {new_ip}",
                                confidence=self._rng.uniform(0.65, 0.90),
                                raw_log=f"ids: Connection from previously-unseen IP {new_ip} to {target_node.ip_address}:443",
                                is_false_positive=False,
                                related_node_id=phase.target_node_id,
                                related_attack_phase=phase.phase_id,
                            ))
                    else:
                        # No more C2 IPs to rotate to — contained
                        phase.is_contained = True
                        self.metrics.threats_neutralized += 1
                        events.append(f"Attack phase '{phase.name}' contained — all C2 IPs exhausted.")
                        continue
                elif blocked:
                    phase.is_contained = True
                    self.metrics.threats_neutralized += 1
                    events.append(f"Attack phase '{phase.name}' contained — attacker IP blocked at firewall.")
                    continue

            # Attacker progresses
            phase.steps_elapsed += 1

            # Handle exfiltration
            if phase.exfiltration_rate_mb > 0 and target_node and target_node.status != NodeStatus.ISOLATED:
                self._total_exfiltrated_mb += phase.exfiltration_rate_mb
                self.metrics.data_exfiltrated_mb = self._total_exfiltrated_mb
                events.append(
                    f"Data exfiltration in progress: {phase.exfiltration_rate_mb} MB stolen this step "
                    f"(total: {self._total_exfiltrated_mb:.1f} MB)"
                )

            # Check if phase completes (attacker achieves objective)
            if phase.steps_elapsed >= phase.steps_to_complete:
                phase.is_completed = True
                # Ransomware encrypts; other attacks compromise
                if target_node and target_node.status == NodeStatus.HEALTHY:
                    if phase.attack_type == AlertType.RANSOMWARE:
                        network.encrypt_node(phase.target_node_id)
                    else:
                        network.compromise_node(phase.target_node_id, network.elapsed_steps())
                events.append(
                    f"⚠️ Attack phase '{phase.name}' COMPLETED — "
                    f"attacker achieved objective on {phase.target_node_id}!"
                )

                # Activate dependent phases
                for next_phase in self.phases:
                    if (next_phase.prerequisite_phase_id == phase.phase_id
                            and not next_phase.is_active):
                        next_phase.is_active = True
                        events.append(f"New attack phase activated: '{next_phase.name}'")

                        # Generate alert for the new phase
                        target = network.get_node(next_phase.target_node_id)
                        if target:
                            alert_id = network.generate_alert_id()
                            network.add_alert(NetworkAlert(
                                alert_id=alert_id,
                                timestamp=float(network.elapsed_steps()),
                                severity=AlertSeverity.HIGH,
                                source_ip=target.ip_address,
                                destination_ip="10.0.1.1",
                                alert_type=next_phase.attack_type,
                                description=f"[HIGH] {next_phase.name}: {next_phase.description}",
                                confidence=self._rng.uniform(0.70, 0.95),
                                raw_log=f"ids: Suspicious activity detected — {next_phase.description}",
                                is_false_positive=False,
                                related_node_id=next_phase.target_node_id,
                                related_attack_phase=next_phase.phase_id,
                            ))

        # Honeypot intel gathering
        if network.honeypot_deployed and any(
            p.is_active and not p.is_contained for p in self.phases
        ):
            intel = f"Honeypot captured attacker reconnaissance at step {network.elapsed_steps()}"
            network.honeypot_intel.append(intel)
            self.metrics.intel_gathered += 0.5
            events.append(f"🍯 {intel}")

        return events

    def is_fully_contained(self) -> bool:
        """Check if all attack phases are resolved (no active threats remain).

        A phase is considered resolved if:
        - It was contained before completion (defender won)
        - It completed AND was subsequently contained (post-compromise containment)
        - It completed but is no longer active (attacker achieved objective, no further action needed)
        - It was never activated (prerequisite not met)
        """
        if not self.phases:
            return True
        return all(
            p.is_contained or p.is_completed or (not p.is_active)
            for p in self.phases
        )

    def get_active_incidents(self) -> list[str]:
        """Get descriptions of currently active attack phases."""
        return [
            f"{p.name} (targeting {p.target_node_id}, progress: {p.steps_elapsed}/{p.steps_to_complete})"
            for p in self.phases
            if p.is_active and not p.is_contained and not p.is_completed
        ]

    def get_state_summary(self) -> dict:
        """Get a summary of attack engine state for the agent."""
        return {
            "active_phases": [
                {
                    "name": p.name,
                    "target": p.target_node_id,
                    "type": p.attack_type.value,
                    "contained": p.is_contained,
                    "completed": p.is_completed,
                }
                for p in self.phases if p.is_active
            ],
            "total_threats": self.metrics.total_threats,
            "threats_neutralized": self.metrics.threats_neutralized,
            "data_exfiltrated_mb": round(self._total_exfiltrated_mb, 1),
        }

    def update_metrics(self, result: "ActionResult") -> None:
        """Update cumulative metrics based on an action result.

        Note: threats_neutralized is tracked in advance() by checking actual
        containment status (node isolated / IP blocked), NOT here. This avoids
        double-counting when both the action result and the advance loop detect
        the same containment event.
        """
        if result.false_positive_correctly_dismissed:
            self.metrics.false_positives_correctly_dismissed += 1
        if result.real_threat_ignored:
            self.metrics.real_threats_ignored += 1
        if result.healthy_host_isolated:
            self.metrics.healthy_hosts_isolated += 1
        if result.critical_services_disrupted:
            self.metrics.critical_services_disrupted += 1
        if result.exfiltration_prevented_mb > 0:
            self.metrics.data_exfiltration_prevented_mb += result.exfiltration_prevented_mb
        if result.attack_chain_resolved:
            self.metrics.attack_chains_resolved += 1
        self.metrics.intel_gathered += result.intel_gathered
        self.metrics.total_resource_cost += result.resource_cost

    # --- Grading ---

    @staticmethod
    def _sanitize_scores(obj: any) -> any:
        """Recursively clamp every numeric value in a nested structure to strictly [0.01, 0.99].

        The OpenEnv validator rejects any task score that is exactly 0.0 or 1.0.
        If the validator rounds to 2 decimal places, 0.001 becomes 0.0 and 0.999 becomes 1.0.
        We strictly bound it well within 0 and 1.
        """
        if isinstance(obj, dict):
            return {k: AttackEngine._sanitize_scores(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [AttackEngine._sanitize_scores(v) for v in obj]
        if isinstance(obj, float):
            if obj < 0.01:
                return 0.01
            if obj > 0.99:
                return 0.99
            return obj
        if isinstance(obj, int) and not isinstance(obj, bool):
            if obj <= 0:
                return 0.01
            if obj >= 1:
                return 0.99
            return float(obj)
        return obj

    def grade_episode(self, network: NetworkSimulator, steps_used: int) -> dict:
        """
        Grade the agent's performance on the current scenario.

        Returns a dict with component scores and a final score (0.0–1.0).
        Grading is deterministic given the same episode outcome.
        """
        if not self.scenario:
            return self._sanitize_scores({"final_score": 0.01, "error": "No scenario loaded"})

        scores: dict[str, float] = {}
        max_steps = self.scenario.max_steps

        # --- Component 1: Threat Neutralization (35%) ---
        if self.metrics.total_threats > 0:
            neutralization_rate = self.metrics.threats_neutralized / self.metrics.total_threats
        else:
            neutralization_rate = 1.0

        # Penalize for attacker completing objectives
        completed_phases = sum(1 for p in self.phases if p.is_completed and not p.is_contained)
        completion_penalty = completed_phases / max(len(self.phases), 1)
        threat_score = max(0.0, neutralization_rate - 0.5 * completion_penalty)
        scores["threat_neutralization"] = round(threat_score * 0.35, 3)

        # --- Component 2: False Positive Handling (20%) ---
        if self.metrics.false_positives_total > 0:
            fp_accuracy = self.metrics.false_positives_correctly_dismissed / self.metrics.false_positives_total
            fp_mistakes = self.metrics.false_positives_incorrectly_acted_on / self.metrics.false_positives_total
            fp_score = fp_accuracy - 0.3 * fp_mistakes
        else:
            fp_score = 1.0  # No FPs to handle = perfect by default
        scores["false_positive_handling"] = round(max(0.0, fp_score) * 0.20, 3)

        # --- Component 3: Data Protection (20%) ---
        total_possible_exfil = sum(
            p.exfiltration_rate_mb * p.steps_to_complete
            for p in self.phases if p.exfiltration_rate_mb > 0
        )
        if total_possible_exfil > 0:
            data_score = 1.0 - min(1.0, self._total_exfiltrated_mb / total_possible_exfil)
        else:
            data_score = 1.0
        scores["data_protection"] = round(data_score * 0.20, 3)

        # --- Component 4: Collateral Damage (15%) ---
        collateral_score = 1.0
        if self.metrics.healthy_hosts_isolated > 0:
            collateral_score -= 0.25 * self.metrics.healthy_hosts_isolated
        if self.metrics.critical_services_disrupted > 0:
            collateral_score -= 0.4 * self.metrics.critical_services_disrupted
        if self.metrics.real_threats_ignored > 0:
            collateral_score -= 0.3 * self.metrics.real_threats_ignored
        scores["collateral_damage"] = round(max(0.0, collateral_score) * 0.15, 3)

        # --- Component 5: Efficiency (10%) ---
        if steps_used <= max_steps * 0.5:
            efficiency_score = 1.0
        elif steps_used <= max_steps * 0.75:
            efficiency_score = 0.7
        elif steps_used <= max_steps:
            efficiency_score = 0.4
        else:
            efficiency_score = 0.1
        scores["efficiency"] = round(efficiency_score * 0.10, 3)

        # --- Final Score ---
        final_score = sum(scores.values())

        # Clamp ALL scores to strictly (0, 1) — evaluator rejects 0.0 and 1.0
        for key in list(scores.keys()):
            scores[key] = round(min(0.99, max(0.01, scores[key])), 4)
        scores["final_score"] = round(min(0.99, max(0.01, final_score)), 4)
        scores["score"] = scores["final_score"]

        # details: ALL numeric values converted to strings so evaluator
        # cannot mistake them for score fields (0 and 1 are integers, not scores)
        scores["details"] = {
            "threats_neutralized": f"{self.metrics.threats_neutralized}/{self.metrics.total_threats}",
            "false_positives_dismissed": f"{self.metrics.false_positives_correctly_dismissed}/{self.metrics.false_positives_total}",
            "data_exfiltrated_mb": f"{round(self._total_exfiltrated_mb, 1)} MB",
            "healthy_hosts_isolated": str(self.metrics.healthy_hosts_isolated),
            "steps_used": f"{steps_used}/{max_steps}",
            "difficulty": self.scenario.difficulty.value,
            "adversary_behavior": self._adversary_behavior,
        }

        # mitre_coverage: convert counts to strings to avoid integer 0/1 issues
        mitre = self.mitre_coverage_report()
        if "total_techniques" in mitre:
            mitre["total_techniques"] = str(mitre["total_techniques"])
        scores["mitre_coverage"] = mitre

        # Final recursive sanitization: catch any remaining numeric edge cases
        # across the entire grader_result tree (including mitre_coverage etc.)
        return self._sanitize_scores(scores)

    def mitre_coverage_report(self) -> dict:
        """
        Generate a MITRE ATT&CK coverage report for the current scenario.

        Returns a dict mapping tactic → list of techniques tested,
        enabling benchmarking against the ATT&CK framework.
        """
        if not self.scenario:
            return {}

        tactics: dict[str, list[dict]] = {}
        for phase in self.phases:
            if not phase.mitre_technique_id:
                continue
            tactic = phase.mitre_tactic or "unknown"
            entry = {
                "technique_id": phase.mitre_technique_id,
                "technique_name": phase.mitre_technique_name,
                "phase": phase.name,
                "target": phase.target_node_id,
                "status": (
                    "contained" if phase.is_contained
                    else "completed" if phase.is_completed
                    else "active" if phase.is_active
                    else "pending"
                ),
            }
            tactics.setdefault(tactic, []).append(entry)

        return {
            "scenario_id": self.scenario.scenario_id,
            "adversary_behavior": self._adversary_behavior,
            "total_techniques": len(self.scenario.mitre_techniques_covered),
            "technique_ids": self.scenario.mitre_techniques_covered,
            "tactics_breakdown": tactics,
        }
