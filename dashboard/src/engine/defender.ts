// heuristic SOC agent — rule-based defender
// same logic as DemoAgent in app_demo.py

import type { Alert, AgentAction, SocPhase } from './types';
import type { NetworkState } from './network';

// tracks what the agent has already done
interface DefenderState {
  step: number;
  investigated: Set<string>;
  blocked: Set<string>;
  dismissed: Set<string>;
  isolated: Set<string>;
  honeypotDeployed: boolean;
  fpCandidates: string[];
  threatIps: string[];
  compromisedNodes: string[];
  currentPhase: SocPhase;
}

export function createDefenderState(): DefenderState {
  return {
    step: 0,
    investigated: new Set(),
    blocked: new Set(),
    dismissed: new Set(),
    isolated: new Set(),
    honeypotDeployed: false,
    fpCandidates: [],
    threatIps: [],
    compromisedNodes: [],
    currentPhase: 'triage',
  };
}

// process the result of the last action to extract intel
export function processEvidence(
  defender: DefenderState,
  alert: Alert | undefined,
): void {
  if (!alert) return;

  const evidence = alert.forensicEvidence.toLowerCase();
  const isBenign = [
    'benign', 'routine', 'scheduled', 'legitimate', 'baseline',
    'no unauthorized', 'appears clean', 'matches expected',
    'nagios', 'health check', 'backup job', 'false positive',
    'normal operation', 'expected behavior', 'cron', 'nessus',
    'windows update', 'defender', 'certbot', 'logrotate',
    'azure ad connect', 'print spooler',
  ].some(keyword => evidence.includes(keyword));

  if (isBenign) {
    if (!defender.fpCandidates.includes(alert.alertId)) {
      defender.fpCandidates.push(alert.alertId);
    }
  } else {
    // real threat — extract IOCs
    if (alert.sourceIp && !alert.sourceIp.startsWith('10.0.') && !defender.blocked.has(alert.sourceIp)) {
      defender.threatIps.push(alert.sourceIp);
    }
    if (alert.targetNodeId && !defender.isolated.has(alert.targetNodeId)) {
      if (!defender.compromisedNodes.includes(alert.targetNodeId)) {
        defender.compromisedNodes.push(alert.targetNodeId);
      }
    }
  }
}

// decide what to do next
export function decide(
  defender: DefenderState,
  alerts: Alert[],
  _networkState: NetworkState,
  difficulty: string,
): AgentAction {
  defender.step++;

  // step 1: always observe first
  if (defender.step === 1) {
    defender.currentPhase = 'triage';
    return {
      tool: 'observe_network',
      args: {},
      reasoning: 'Starting incident response. Scanning full network state.',
    };
  }

  // deploy honeypot early on hard+ scenarios
  if (!defender.honeypotDeployed && (difficulty === 'hard' || difficulty === 'nightmare') && defender.step <= 3) {
    defender.honeypotDeployed = true;
    return {
      tool: 'deploy_honeypot',
      args: {},
      reasoning: 'Deploying honeypot for adversary intelligence gathering.',
    };
  }

  // investigate uninvestigated alerts (sorted by severity)
  const severityOrder: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3, info: 4 };
  const uninvestigated = alerts
    .filter(a => !defender.investigated.has(a.alertId))
    .sort((a, b) => (severityOrder[a.severity] ?? 4) - (severityOrder[b.severity] ?? 4));

  if (uninvestigated.length > 0) {
    const alert = uninvestigated[0];
    defender.investigated.add(alert.alertId);
    defender.currentPhase = 'investigate';

    // process the evidence from this alert
    processEvidence(defender, alert);

    return {
      tool: 'investigate_alert',
      args: { alert_id: alert.alertId },
      reasoning: `Investigating [${alert.severity.toUpperCase()}] alert ${alert.alertId}: ${alert.title}`,
    };
  }

  // block known threat IPs
  if (defender.threatIps.length > 0) {
    const ip = defender.threatIps.shift()!;
    if (!defender.blocked.has(ip)) {
      defender.blocked.add(ip);
      defender.currentPhase = 'contain';
      return {
        tool: 'block_ip',
        args: { ip_address: ip },
        reasoning: `Blocking attacker IP ${ip} at perimeter firewall.`,
      };
    }
  }

  // block well-known malicious IPs from scenario data
  const knownBadIps = ['185.220.101.42', '94.232.46.19', '45.155.205.233', '91.219.236.166', '198.51.100.88', '203.0.113.45'];
  for (const ip of knownBadIps) {
    if (!defender.blocked.has(ip) && alerts.some(a => a.sourceIp === ip && !a.isFalsePositive)) {
      defender.blocked.add(ip);
      defender.currentPhase = 'contain';
      return {
        tool: 'block_ip',
        args: { ip_address: ip },
        reasoning: `Proactively blocking known-malicious IP ${ip}.`,
      };
    }
  }

  // dismiss confirmed false positives
  if (defender.fpCandidates.length > 0) {
    const alertId = defender.fpCandidates.shift()!;
    if (!defender.dismissed.has(alertId)) {
      defender.dismissed.add(alertId);
      defender.currentPhase = 'investigate';
      return {
        tool: 'dismiss_alert',
        args: { alert_id: alertId },
        reasoning: `Evidence confirms benign activity. Dismissing ${alertId} as false positive.`,
      };
    }
  }

  // isolate compromised nodes
  if (defender.compromisedNodes.length > 0) {
    const nodeId = defender.compromisedNodes.shift()!;
    if (!defender.isolated.has(nodeId)) {
      defender.isolated.add(nodeId);
      defender.currentPhase = 'contain';
      return {
        tool: 'isolate_host',
        args: { node_id: nodeId },
        reasoning: `Isolating confirmed-compromised host ${nodeId} from network.`,
      };
    }
  }

  // fallback: observe for new threats
  defender.currentPhase = 'remediate';
  return {
    tool: 'observe_network',
    args: {},
    reasoning: 'All known threats addressed. Scanning for new activity.',
  };
}
