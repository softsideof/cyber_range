// attack progression logic
// ported from attack_engine.py — handles adversary behavior and phase advancement

import type { AttackPhase, Alert, AdversaryBehavior } from './types';
import type { NetworkState } from './network';
import { getNode } from './network';

// backup C2 IPs the evasive adversary rotates to
const BACKUP_C2_IPS = [
  '198.51.100.23',
  '203.0.113.77',
  '198.51.100.99',
  '45.155.205.100',
  '91.219.236.200',
];

// advance all active attack phases by one step
export function progressAttacks(
  phases: AttackPhase[],
  networkState: NetworkState,
  adversaryBehavior: AdversaryBehavior,
  currentStep: number,
): { updatedPhases: AttackPhase[]; newAlerts: Alert[]; events: string[] } {
  const events: string[] = [];
  const newAlerts: Alert[] = [];

  for (const phase of phases) {
    if (!phase.isActive || phase.isNeutralized) continue;

    // check if this phase's source IP is blocked
    const ipBlocked = networkState.blockedIps.has(phase.sourceIp);

    // check if target node is isolated
    const targetIsolated = networkState.isolatedNodes.has(phase.targetNodeId);

    if (targetIsolated) {
      phase.isNeutralized = true;
      events.push(`Phase "${phase.name}" neutralized — target ${phase.targetNodeId} is isolated.`);
      continue;
    }

    if (ipBlocked) {
      if (adversaryBehavior === 'evasive' || adversaryBehavior === 'adaptive') {
        // rotate to backup C2 IP
        const newIp = BACKUP_C2_IPS.find(ip => !networkState.blockedIps.has(ip));
        if (newIp) {
          phase.sourceIp = newIp;
          events.push(`Adversary rotated C2 from blocked IP to ${newIp}.`);
        } else {
          phase.isNeutralized = true;
          events.push(`Phase "${phase.name}" neutralized — all C2 IPs blocked.`);
          continue;
        }
      } else {
        phase.isNeutralized = true;
        events.push(`Phase "${phase.name}" neutralized — source IP blocked.`);
        continue;
      }
    }

    // advance the phase
    phase.stepsElapsed++;

    // check if phase completes (target gets compromised)
    if (phase.stepsElapsed >= phase.stepsToComplete) {
      const targetNode = getNode(networkState, phase.targetNodeId);
      if (targetNode && targetNode.status !== 'isolated') {
        targetNode.status = phase.compromiseEffect === 'encrypted' ? 'encrypted' : 'compromised';
        events.push(`Phase "${phase.name}" completed — ${phase.targetNodeId} is now ${targetNode.status}.`);
      }
      phase.isActive = false;
    }
  }

  // persistent adversary re-compromises patched (not restored) nodes
  if (adversaryBehavior === 'persistent' || adversaryBehavior === 'adaptive') {
    for (const phase of phases) {
      if (!phase.isNeutralized) continue;

      const node = getNode(networkState, phase.targetNodeId);
      if (node && node.status === 'patched') {
        // patched but not restored — adversary still has backdoor
        node.status = 'compromised';
        phase.isNeutralized = false;
        phase.isActive = true;
        phase.stepsElapsed = Math.max(0, phase.stepsToComplete - 3);
        events.push(`Persistent adversary re-compromised ${phase.targetNodeId} — patch insufficient, needs full restore.`);
      }
    }
  }

  // adaptive adversary injects decoy alerts mid-episode
  if (adversaryBehavior === 'adaptive' && currentStep % 8 === 0 && currentStep > 0) {
    newAlerts.push({
      alertId: `DECOY-${currentStep}`,
      severity: 'medium',
      type: 'anomalous_traffic',
      title: 'Suspicious Network Scan',
      sourceIp: '10.0.4.' + (Math.floor(Math.random() * 4) + 1),
      targetNodeId: ['ws-01', 'ws-02', 'ws-03', 'app-01'][Math.floor(Math.random() * 4)],
      description: 'Network scan activity detected from internal host.',
      timestamp: currentStep,
      status: 'new',
      isFalsePositive: true,
      forensicEvidence: 'Benign. Scheduled Nessus vulnerability scan initiated by IT security team. Matches approved scan window.',
      mitreId: '',
      confidence: 0.3,
    });
    events.push('Adaptive adversary injected decoy alert.');
  }

  return { updatedPhases: phases, newAlerts, events };
}

// check if a specific phase has been neutralized
export function isPhaseNeutralized(
  phase: AttackPhase,
  networkState: NetworkState,
): boolean {
  if (phase.isNeutralized) return true;
  if (networkState.isolatedNodes.has(phase.targetNodeId)) return true;
  if (networkState.blockedIps.has(phase.sourceIp)) return true;
  return false;
}

// count how many phases are still active (not neutralized, not completed)
export function countActiveThreats(phases: AttackPhase[]): number {
  return phases.filter(p => p.isActive && !p.isNeutralized).length;
}

// count neutralized phases
export function countNeutralizedThreats(phases: AttackPhase[]): number {
  return phases.filter(p => p.isNeutralized).length;
}
