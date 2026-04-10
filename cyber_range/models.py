"""Internal simulation models (enums, dataclasses, typed dicts)."""


from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, TypedDict


class ToolResponse(TypedDict, total=False):
    """Standardized response from all MCP analytical tools."""
    action: str
    result: str
    success: bool
    reward: float
    network_summary: dict[str, Any]
    details: dict[str, Any]
    threat_neutralized: bool
    business_disruption: bool
    intel_gathered: str
    cost: float



class NodeType(str, Enum):
    """Types of hosts on the enterprise network."""
    FIREWALL = "firewall"
    DOMAIN_CONTROLLER = "domain_controller"
    WEB_SERVER = "web_server"
    MAIL_SERVER = "mail_server"
    DATABASE = "database"
    APP_SERVER = "app_server"
    WORKSTATION = "workstation"
    HONEYPOT = "honeypot"
    BACKUP_SERVER = "backup_server"


class NodeStatus(str, Enum):
    """Operational status of a network host."""
    HEALTHY = "healthy"
    COMPROMISED = "compromised"
    ISOLATED = "isolated"
    OFFLINE = "offline"
    ENCRYPTED = "encrypted"  # Ransomware


class AlertSeverity(str, Enum):
    """SIEM alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(str, Enum):
    """Categories of security alerts."""
    INTRUSION = "intrusion"
    MALWARE = "malware"
    EXFILTRATION = "exfiltration"
    LATERAL_MOVEMENT = "lateral_movement"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    BRUTE_FORCE = "brute_force"
    PHISHING = "phishing"
    RANSOMWARE = "ransomware"
    ANOMALOUS_TRAFFIC = "anomalous_traffic"


class ThreatLevel(str, Enum):
    """Overall network threat assessment."""
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"
    CRITICAL = "critical"


class Difficulty(str, Enum):
    """Scenario difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    NIGHTMARE = "nightmare"


class AdversaryBehavior(str, Enum):
    """Adversary adaptation strategy."""
    STATIC = "static"          # Fixed attack plan, no adaptation
    EVASIVE = "evasive"        # Rotates C2 IPs when blocked, changes TTPs
    PERSISTENT = "persistent"  # Re-compromises hosts if only patched (not restored)
    ADAPTIVE = "adaptive"      # Full adaptation: evasion + persistence + decoy alerts


class MitreTactic(str, Enum):
    """MITRE ATT&CK Tactic categories."""
    INITIAL_ACCESS = "initial-access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege-escalation"
    DEFENSE_EVASION = "defense-evasion"
    CREDENTIAL_ACCESS = "credential-access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral-movement"
    COLLECTION = "collection"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"



@dataclass
class NetworkNode:
    """A host on the simulated enterprise network."""
    node_id: str
    hostname: str
    ip_address: str
    node_type: NodeType
    os: str
    status: NodeStatus = NodeStatus.HEALTHY
    open_ports: list[int] = field(default_factory=list)
    running_services: list[str] = field(default_factory=list)
    is_critical: bool = False
    vulnerabilities: list[str] = field(default_factory=list)
    compromised_at_step: int = -1
    isolated_at_step: int = -1

    def to_dict(self) -> dict:
        """Serialize to dict for agent observation."""
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "node_type": self.node_type.value,
            "os": self.os,
            "status": self.status.value,
            "open_ports": self.open_ports,
            "running_services": self.running_services,
            "is_critical": self.is_critical,
        }


@dataclass
class NetworkAlert:
    """A security alert from the SIEM system."""
    alert_id: str
    timestamp: float
    severity: AlertSeverity
    source_ip: str
    destination_ip: str
    alert_type: AlertType
    description: str
    confidence: float  # 0.0–1.0 (some alerts are false positives!)
    raw_log: str
    is_false_positive: bool = False
    related_node_id: str = ""
    related_attack_phase: str = ""
    investigated: bool = False
    resolved: bool = False
    dismissed: bool = False

    def to_dict(self) -> dict:
        """Serialize to dict for agent observation (hides ground truth)."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "source_ip": self.source_ip,
            "destination_ip": self.destination_ip,
            "alert_type": self.alert_type.value,
            "description": self.description,
            "confidence": self.confidence,
            "raw_log": self.raw_log,
            "investigated": self.investigated,
            "resolved": self.resolved,
            "dismissed": self.dismissed,
        }


@dataclass
class AttackPhase:
    """A single phase in a multi-step attack chain, aligned to MITRE ATT&CK."""
    phase_id: str
    name: str
    description: str
    target_node_id: str
    attack_type: AlertType
    steps_to_complete: int  # Steps before attacker achieves objective
    steps_elapsed: int = 0
    exfiltration_rate_mb: float = 0.0  # MB per step if exfiltrating
    is_active: bool = False
    is_completed: bool = False  # Attacker succeeded
    is_contained: bool = False  # Defender neutralized
    prerequisite_phase_id: Optional[str] = None  # Must complete before this activates
    # MITRE ATT&CK alignment
    mitre_technique_id: str = ""       # e.g. "T1190"
    mitre_technique_name: str = ""     # e.g. "Exploit Public-Facing Application"
    mitre_tactic: str = ""             # e.g. "initial-access"
    # Adaptive adversary fields
    c2_ip_pool: list[str] = field(default_factory=list)  # Backup C2 IPs for rotation
    recompromise_delay: int = 0        # Steps before re-compromise if only patched
    times_recompromised: int = 0       # Track persistence attempts


@dataclass
class ForensicArtifact:
    """Structured forensic evidence from host analysis."""
    process_tree: list[dict] = field(default_factory=list)       # Parent→child process chains
    network_connections: list[dict] = field(default_factory=list) # Active connections with ports
    registry_modifications: list[str] = field(default_factory=list)  # Windows registry changes
    file_artifacts: list[dict] = field(default_factory=list)     # Modified files with hashes
    memory_indicators: list[str] = field(default_factory=list)   # IOCs from memory analysis
    mitre_techniques_observed: list[str] = field(default_factory=list)  # Detected ATT&CK techniques


@dataclass
class ScenarioConfig:
    """A complete attack scenario with grading criteria."""
    scenario_id: str
    name: str
    description: str
    difficulty: Difficulty
    threat_count: int
    false_positive_count: int
    max_steps: int
    attack_phases: list[AttackPhase] = field(default_factory=list)
    initial_compromised_nodes: list[str] = field(default_factory=list)
    adversary_behavior: str = "static"  # AdversaryBehavior value
    mitre_techniques_covered: list[str] = field(default_factory=list)  # All MITRE IDs in scenario


@dataclass
class ActionResult:
    """Result of executing an agent's defensive action."""
    action_type: str
    success: bool
    description: str
    threat_neutralized: bool = False
    threat_severity_multiplier: float = 1.0
    false_positive_correctly_dismissed: bool = False
    real_threat_ignored: bool = False
    healthy_host_isolated: bool = False
    exfiltration_prevented_mb: float = 0.0
    intel_gathered: float = 0.0
    resource_cost: float = 0.0
    critical_services_disrupted: bool = False
    health_delta: float = 0.0
    attack_chain_resolved: bool = False
    details: dict = field(default_factory=dict)


@dataclass
class EpisodeMetrics:
    """Cumulative metrics tracked throughout an episode."""
    total_threats: int = 0
    threats_neutralized: int = 0
    false_positives_total: int = 0
    false_positives_correctly_dismissed: int = 0
    false_positives_incorrectly_acted_on: int = 0
    real_threats_ignored: int = 0
    healthy_hosts_isolated: int = 0
    data_exfiltrated_mb: float = 0.0
    data_exfiltration_prevented_mb: float = 0.0
    critical_services_disrupted: int = 0
    total_resource_cost: float = 0.0
    attack_chains_resolved: int = 0
    total_attack_chains: int = 0
    intel_gathered: float = 0.0
