// types for the simulation engine
// mirrors the python dataclasses in cyber_range/models.py

export type NodeType =
  | 'firewall'
  | 'domain_controller'
  | 'web_server'
  | 'mail_server'
  | 'database'
  | 'app_server'
  | 'workstation'
  | 'honeypot'
  | 'backup_server';

export type NodeStatus =
  | 'healthy'
  | 'compromised'
  | 'isolated'
  | 'offline'
  | 'encrypted'
  | 'patched';

export type AlertSeverity = 'critical' | 'high' | 'medium' | 'low' | 'info';

export type AlertType =
  | 'intrusion'
  | 'malware'
  | 'exfiltration'
  | 'lateral_movement'
  | 'privilege_escalation'
  | 'brute_force'
  | 'phishing'
  | 'ransomware'
  | 'anomalous_traffic';

export type ThreatLevel = 'green' | 'yellow' | 'orange' | 'red' | 'critical';

export type Difficulty = 'easy' | 'medium' | 'hard' | 'nightmare';

export type AdversaryBehavior = 'static' | 'evasive' | 'persistent' | 'adaptive';

export type AlertStatus = 'new' | 'investigating' | 'contained' | 'dismissed';

export type SocPhase = 'triage' | 'investigate' | 'contain' | 'remediate';

// network node in the 12-node enterprise topology
export interface NetworkNode {
  nodeId: string;
  hostname: string;
  ip: string;
  type: NodeType;
  os: string;
  status: NodeStatus;
  openPorts: number[];
  services: string[];
  isCritical: boolean;
  vulnerabilities: string[];
}

// siem alert from the network
export interface Alert {
  alertId: string;
  severity: AlertSeverity;
  type: AlertType;
  title: string;
  sourceIp: string;
  targetNodeId: string;
  description: string;
  timestamp: number; // step number when alert fired
  status: AlertStatus;
  isFalsePositive: boolean;
  forensicEvidence: string;
  mitreId: string;
  confidence: number; // 0-1, low confidence = likely FP
}

// single phase in a multi-stage attack
export interface AttackPhase {
  phaseId: string;
  name: string;
  description: string;
  targetNodeId: string;
  attackType: AlertType;
  stepsToComplete: number;
  stepsElapsed: number;
  isActive: boolean;
  isNeutralized: boolean;
  mitreId: string;
  mitreName: string;
  mitreTactic: string;
  // what happens if uncontained
  compromiseEffect: 'compromised' | 'encrypted' | 'exfiltrated';
  sourceIp: string;
}

// full scenario definition
export interface ScenarioConfig {
  id: string;
  name: string;
  description: string;
  difficulty: Difficulty;
  maxSteps: number;
  adversaryBehavior: AdversaryBehavior;
  mitreTechniques: string[];
  attackPhases: AttackPhase[];
  alerts: Alert[];
  falsePositiveAlerts: Alert[];
  initialCompromisedNodes: string[];
  threatCount: number;
  falsePositiveCount: number;
}

// agent action
export interface AgentAction {
  tool: string;
  args: Record<string, string>;
  reasoning: string;
}

// result of an agent action
export interface ActionResult {
  success: boolean;
  description: string;
  cost: number;
  reward: number;
  details: Record<string, unknown>;
}

// one entry in the agent log
export interface AgentLogEntry {
  step: number;
  action: AgentAction;
  result: ActionResult;
  reward: number;
  phase: SocPhase;
  topologySnapshot: NodeStatus[]; // just the statuses, indexed by node order
}

// live metrics during simulation
export interface SimMetrics {
  health: number;       // 0-100
  threatLevel: ThreatLevel;
  budget: number;       // remaining action budget
  maxBudget: number;
  dataAtRiskMb: number; // potential data exfiltration
  score: number;        // running episode score
}

// mitre technique tracking
export interface MitreEntry {
  id: string;
  name: string;
  tactic: string;
  state: 'inactive' | 'attacking' | 'defended';
}

// final grading result
export interface GraderResult {
  finalScore: number;
  threatResponse: number;
  falsePositiveHandling: number;
  dataProtection: number;
  collateralDamage: number;
  efficiency: number;
  details: {
    threatsNeutralized: number;
    totalThreats: number;
    fpDismissed: number;
    totalFps: number;
    stepsUsed: number;
    maxSteps: number;
    dataExfiltratedMb: number;
    healthyHostsIsolated: number;
    adversaryBehavior: string;
  };
  judgeVerdicts: {
    junior: { score: number; verdict: string };
    senior: { score: number; verdict: string };
    commander: { score: number; verdict: string };
  };
}

// full simulation state sent from worker to main thread
export interface SimulationState {
  mode: 'idle' | 'briefing' | 'running' | 'scoring' | 'complete';
  scenario: ScenarioConfig | null;
  step: number;
  maxSteps: number;
  topology: NetworkNode[];
  alerts: Alert[];
  agentLog: AgentLogEntry[];
  metrics: SimMetrics;
  mitre: MitreEntry[];
  graderResult: GraderResult | null;
  currentPhase: SocPhase;
}

// messages between main thread and worker
export type WorkerCommand =
  | { type: 'LAUNCH'; scenarioId: string }
  | { type: 'LAUNCH_CUSTOM'; config: CustomAttackConfig }
  | { type: 'PAUSE' }
  | { type: 'RESUME' }
  | { type: 'SET_SPEED'; speed: number }
  | { type: 'STEP_ONCE' }
  | { type: 'RESET' }
  | { type: 'STOP' };

export type WorkerMessage =
  | { type: 'STATE_UPDATE'; state: SimulationState }
  | { type: 'EPISODE_END'; state: SimulationState }
  | { type: 'ERROR'; message: string };

// custom attack config from the attack builder UI
export interface CustomAttackConfig {
  name: string;
  description: string;
  attackVector: AlertType;
  targets: string[];
  evasion: {
    rotateC2: boolean;
    recompromise: boolean;
    decoyAlerts: boolean;
  };
  difficulty: Difficulty;
  maxSteps: number;
  phases: Array<{
    name: string;
    target: string;
    type: AlertType;
    mitreId: string;
  }>;
}
