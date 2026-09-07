// custom attack builder & pre-built virus templates
// converts user-designed attacks or templates into full ScenarioConfigs

import type {
  ScenarioConfig,
  AttackPhase,
  Alert,
  CustomAttackConfig,
  AlertType,
  AdversaryBehavior,
  Difficulty,
} from './types';

// mitre technique reference metadata for auto-enrichment
interface MitreMetadata {
  id: string;
  name: string;
  tactic: string;
}

const MITRE_LOOKUP: Record<string, MitreMetadata> = {
  T1190: { id: 'T1190', name: 'Exploit Public-Facing Application', tactic: 'Initial Access' },
  T1566: { id: 'T1566', name: 'Phishing', tactic: 'Initial Access' },
  T1195: { id: 'T1195', name: 'Supply Chain Compromise', tactic: 'Initial Access' },
  T1078: { id: 'T1078', name: 'Valid Accounts', tactic: 'Defense Evasion' },
  T1059: { id: 'T1059', name: 'Command & Scripting Interpreter', tactic: 'Execution' },
  T1204: { id: 'T1204', name: 'User Execution', tactic: 'Execution' },
  T1003: { id: 'T1003', name: 'OS Credential Dumping', tactic: 'Credential Access' },
  T1110: { id: 'T1110', name: 'Brute Force', tactic: 'Credential Access' },
  T1021: { id: 'T1021', name: 'Remote Services (SMB/RDP/SSH)', tactic: 'Lateral Movement' },
  T1074: { id: 'T1074', name: 'Data Staged', tactic: 'Collection' },
  T1041: { id: 'T1041', name: 'Exfiltration Over C2', tactic: 'Exfiltration' },
  T1567: { id: 'T1567', name: 'Exfiltration to Cloud Storage', tactic: 'Exfiltration' },
  T1486: { id: 'T1486', name: 'Data Encrypted for Impact', tactic: 'Impact' },
  T1490: { id: 'T1490', name: 'Inhibit System Recovery', tactic: 'Impact' },
};

// pre-built virus templates ready for interviewer to test
export const VIRUS_TEMPLATES: Record<string, CustomAttackConfig> = {
  wannacry_lite: {
    name: 'WannaCry Lite (Ransomware Outbreak)',
    description: 'Rapid worm outbreak spreading via SMB vulnerability (EternalBlue style), targeting endpoints and disabling backups.',
    attackVector: 'ransomware',
    targets: ['ws-01', 'ws-02', 'ws-03', 'backup-01'],
    evasion: {
      rotateC2: false,
      recompromise: true,
      decoyAlerts: false,
    },
    difficulty: 'hard',
    maxSteps: 25,
    phases: [
      { name: 'Initial Weaponized Attachment', target: 'ws-01', type: 'phishing', mitreId: 'T1566' },
      { name: 'SMB Worm Lateral Spread', target: 'ws-02', type: 'lateral_movement', mitreId: 'T1021' },
      { name: 'Secondary Host Infection', target: 'ws-03', type: 'lateral_movement', mitreId: 'T1021' },
      { name: 'Backup Server Cryptolocker', target: 'backup-01', type: 'ransomware', mitreId: 'T1486' },
    ],
  },
  solarwinds_jr: {
    name: 'SolarWinds Jr. (Supply Chain Trojan)',
    description: 'Stealthy backdoor planted in app server via vendor supply chain, quietly pivoting toward the primary database and domain controller.',
    attackVector: 'malware',
    targets: ['app-01', 'dc-01', 'db-01'],
    evasion: {
      rotateC2: true,
      recompromise: true,
      decoyAlerts: true,
    },
    difficulty: 'hard',
    maxSteps: 30,
    phases: [
      { name: 'Compromised Dependency Artifact', target: 'app-01', type: 'malware', mitreId: 'T1195' },
      { name: 'Domain Controller Kerberoasting', target: 'dc-01', type: 'privilege_escalation', mitreId: 'T1003' },
      { name: 'Database Dumping & Staging', target: 'db-01', type: 'exfiltration', mitreId: 'T1041' },
    ],
  },
  ghost_operator: {
    name: 'Ghost Operator (Dual-Pronged APT & Insider)',
    description: 'Advanced persistent threat coordinating with rogue insider credentials while pivoting externally against web and mail servers.',
    attackVector: 'exfiltration',
    targets: ['ws-04', 'mail-01', 'dc-01', 'db-01'],
    evasion: {
      rotateC2: true,
      recompromise: true,
      decoyAlerts: true,
    },
    difficulty: 'nightmare',
    maxSteps: 40,
    phases: [
      { name: 'Executive Token Abuse', target: 'ws-04', type: 'privilege_escalation', mitreId: 'T1078' },
      { name: 'Cloud Storage Direct Exfil', target: 'ws-04', type: 'exfiltration', mitreId: 'T1567' },
      { name: 'External Exploit on Mail Relay', target: 'mail-01', type: 'intrusion', mitreId: 'T1190' },
      { name: 'Active Directory Shadow Copy Theft', target: 'dc-01', type: 'privilege_escalation', mitreId: 'T1003' },
      { name: 'Production Database Infiltration', target: 'db-01', type: 'lateral_movement', mitreId: 'T1021' },
    ],
  },
};

// pool of realistic false positive alerts to inject based on difficulty
const FP_ALERT_POOL: Alert[] = [
  {
    alertId: 'FP-NAGIOS-01',
    severity: 'low',
    type: 'anomalous_traffic',
    title: 'High Volume Ping Sweep',
    sourceIp: '10.0.3.5',
    targetNodeId: 'web-01',
    description: 'Rapid ICMP echo requests detected targeting web server cluster.',
    timestamp: 2,
    status: 'new',
    isFalsePositive: true,
    forensicEvidence: 'Benign. Routine Nagios infrastructure monitoring health check baseline verified.',
    mitreId: '',
    confidence: 0.2,
  },
  {
    alertId: 'FP-BACKUP-02',
    severity: 'medium',
    type: 'anomalous_traffic',
    title: 'High Outbound Network Spike',
    sourceIp: '10.0.3.1',
    targetNodeId: 'backup-01',
    description: 'PostgreSQL binary replication streaming 12GB of encrypted snapshot data.',
    timestamp: 4,
    status: 'new',
    isFalsePositive: true,
    forensicEvidence: 'Benign. Scheduled nightly postgresql backup job matches crontab entry /etc/cron.d/db-backup.',
    mitreId: '',
    confidence: 0.25,
  },
  {
    alertId: 'FP-WINUPDATE-03',
    severity: 'low',
    type: 'privilege_escalation',
    title: 'SYSTEM Token Impersonation',
    sourceIp: '10.0.4.2',
    targetNodeId: 'ws-02',
    description: 'TrustedInstaller spawning worker processes with elevated NT AUTHORITY\\SYSTEM tokens.',
    timestamp: 5,
    status: 'new',
    isFalsePositive: true,
    forensicEvidence: 'Benign. Legitimate Windows Update cumulative patch installation verified via CBS.log.',
    mitreId: '',
    confidence: 0.15,
  },
  {
    alertId: 'FP-CERTBOT-04',
    severity: 'low',
    type: 'anomalous_traffic',
    title: 'ACME Challenge Inbound',
    sourceIp: '198.51.100.12',
    targetNodeId: 'web-01',
    description: 'Rapid HTTP validation hits on /.well-known/acme-challenge path.',
    timestamp: 7,
    status: 'new',
    isFalsePositive: true,
    forensicEvidence: 'Benign. LetsEncrypt certbot SSL renewal cron job verified with matching challenge tokens.',
    mitreId: '',
    confidence: 0.1,
  },
  {
    alertId: 'FP-NESSUS-05',
    severity: 'medium',
    type: 'brute_force',
    title: 'Rapid Port Enumeration',
    sourceIp: '10.0.0.15',
    targetNodeId: 'dc-01',
    description: 'Over 400 TCP SYN packets per second received across ports 88, 389, 445.',
    timestamp: 9,
    status: 'new',
    isFalsePositive: true,
    forensicEvidence: 'Benign. Authorized Nessus vulnerability assessment scan initiated by sec-ops staff ticket SEC-9402.',
    mitreId: '',
    confidence: 0.3,
  },
];

// builds a full ScenarioConfig from custom user configuration
export function buildCustomScenario(config: CustomAttackConfig): ScenarioConfig {
  // determine adversary behavior from evasion flags
  let adversaryBehavior: AdversaryBehavior = 'static';
  if (config.evasion.rotateC2 && config.evasion.recompromise && config.evasion.decoyAlerts) {
    adversaryBehavior = 'adaptive';
  } else if (config.evasion.recompromise) {
    adversaryBehavior = 'persistent';
  } else if (config.evasion.rotateC2) {
    adversaryBehavior = 'evasive';
  }

  const mitreTechniquesSet = new Set<string>();
  const attackPhases: AttackPhase[] = [];
  const alerts: Alert[] = [];

  const externalIpPool = [
    '198.51.100.77',
    '203.0.113.88',
    '45.155.205.112',
    '91.219.236.44',
    '185.220.101.55',
  ];

  // build each attack phase and matching real threat alert
  config.phases.forEach((phaseDef, idx) => {
    const mitreMeta = MITRE_LOOKUP[phaseDef.mitreId] ?? {
      id: phaseDef.mitreId || 'T1190',
      name: 'Exploit Technique',
      tactic: 'Execution',
    };
    mitreTechniquesSet.add(mitreMeta.id);

    const sourceIp = idx === 0 || config.evasion.rotateC2
      ? externalIpPool[idx % externalIpPool.length]
      : externalIpPool[0];

    const phaseId = `PHASE-CUST-${idx + 1}`;
    const stepsToComplete = Math.max(3, Math.min(8, Math.floor(config.maxSteps / (config.phases.length + 1))));

    // determine compromise effect
    let compromiseEffect: 'compromised' | 'encrypted' | 'exfiltrated' = 'compromised';
    if (phaseDef.type === 'ransomware') {
      compromiseEffect = 'encrypted';
    } else if (phaseDef.type === 'exfiltration') {
      compromiseEffect = 'exfiltrated';
    }

    attackPhases.push({
      phaseId,
      name: phaseDef.name,
      description: `Stage ${idx + 1}: Adversary deploying ${phaseDef.type} on ${phaseDef.target}.`,
      targetNodeId: phaseDef.target,
      attackType: phaseDef.type,
      stepsToComplete,
      stepsElapsed: 0,
      isActive: idx === 0, // first phase starts active
      isNeutralized: false,
      mitreId: mitreMeta.id,
      mitreName: mitreMeta.name,
      mitreTactic: mitreMeta.tactic,
      compromiseEffect,
      sourceIp,
    });

    // matching threat alert
    const sev = (idx === 0 ? 'high' : idx === config.phases.length - 1 ? 'critical' : 'medium') as Alert['severity'];
    alerts.push({
      alertId: `ALT-CUST-${idx + 1}`,
      severity: sev,
      type: phaseDef.type,
      title: `${phaseDef.name} (${phaseDef.target})`,
      sourceIp,
      targetNodeId: phaseDef.target,
      description: `Active intrusion indicator detected on ${phaseDef.target} associated with ${mitreMeta.name}.`,
      timestamp: idx * 3 + 1,
      status: 'new',
      isFalsePositive: false,
      forensicEvidence: `Malicious payload detected: reverse shell binding on port 4444 connecting to ${sourceIp}. Process binary hash matches known adversary campaign.`,
      mitreId: mitreMeta.id,
      confidence: 0.95,
    });
  });

  // select number of false positives according to difficulty
  const fpCount = config.difficulty === 'nightmare' ? 5 : config.difficulty === 'hard' ? 4 : config.difficulty === 'medium' ? 3 : 2;
  const falsePositiveAlerts = FP_ALERT_POOL.slice(0, fpCount).map((fp, i) => ({
    ...fp,
    alertId: `FP-GEN-${i + 1}`,
  }));

  const initialCompromised = config.targets.length > 0 && config.difficulty === 'nightmare'
    ? [config.targets[0]]
    : [];

  return {
    id: `custom-${Date.now()}`,
    name: config.name || 'Custom Attack Simulation',
    description: config.description || 'User-designed adversarial attack simulation.',
    difficulty: config.difficulty,
    maxSteps: config.maxSteps || 25,
    adversaryBehavior,
    mitreTechniques: Array.from(mitreTechniquesSet),
    attackPhases,
    alerts,
    falsePositiveAlerts,
    initialCompromisedNodes: initialCompromised,
    threatCount: attackPhases.length,
    falsePositiveCount: falsePositiveAlerts.length,
  };
}
