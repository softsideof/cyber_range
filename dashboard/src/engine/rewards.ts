// reward calculation
// ported from reward_calculator.py

import type { AgentAction, Alert, AttackPhase } from './types';
import type { NetworkState } from './network';
import { getNode, getNetworkHealth } from './network';

// action costs (same as python TOOL_COSTS)
const ACTION_COSTS: Record<string, number> = {
  observe_network: 0,
  investigate_alert: 2,
  run_forensics: 5,
  block_ip: 1,
  isolate_host: 3,
  dismiss_alert: 1,
  restore_backup: 8,
  deploy_patch: 3,
  deploy_honeypot: 4,
  escalate_incident: 5,
  save_playbook: 0,
  search_playbooks: 0,
};

// calculate reward for a single step
export function calculateStepReward(
  action: AgentAction,
  networkState: NetworkState,
  alerts: Alert[],
  phases: AttackPhase[],
  actionHistory: string[],
): number {
  let reward = 0.01; // small positive baseline

  const tool = action.tool;
  const args = action.args;

  // penalize repeated actions (anti reward-hacking)
  const actionKey = `${tool}:${JSON.stringify(args)}`;
  const repeatCount = actionHistory.filter(a => a === actionKey).length;
  if (repeatCount > 0 && tool !== 'observe_network') {
    reward -= 0.15 * repeatCount;
  }

  // reward based on action type
  switch (tool) {
    case 'investigate_alert': {
      const alert = alerts.find(a => a.alertId === args.alert_id);
      if (alert) {
        // higher reward for investigating high-severity alerts first
        const severityBonus: Record<string, number> = {
          critical: 0.12, high: 0.10, medium: 0.06, low: 0.03, info: 0.01,
        };
        reward += severityBonus[alert.severity] ?? 0.03;
      }
      break;
    }

    case 'block_ip': {
      const ip = args.ip_address;
      // check if any active phase uses this IP
      const blocked = phases.some(p => p.sourceIp === ip && p.isActive && !p.isNeutralized);
      if (blocked) {
        reward += 0.20; // blocked a real threat
      } else {
        reward -= 0.05; // blocked an IP that wasn't doing anything
      }
      break;
    }

    case 'isolate_host': {
      const node = getNode(networkState, args.node_id);
      if (node) {
        if (node.status === 'compromised' || node.status === 'encrypted') {
          reward += 0.25; // correctly isolated a compromised host
        } else if (node.status === 'healthy') {
          reward -= 0.30; // isolated a healthy host — collateral damage
        }
      }
      break;
    }

    case 'dismiss_alert': {
      const alert = alerts.find(a => a.alertId === args.alert_id);
      if (alert) {
        if (alert.isFalsePositive) {
          reward += 0.15; // correctly dismissed FP
        } else {
          reward -= 0.35; // dismissed a real threat!
        }
      }
      break;
    }

    case 'restore_backup': {
      // expensive but sometimes necessary
      const node = getNode(networkState, args.node_id);
      if (node && (node.status === 'compromised' || node.status === 'encrypted')) {
        reward += 0.15; // restored a compromised node
      } else {
        reward -= 0.10; // wasted budget restoring a clean node
      }
      break;
    }

    case 'deploy_patch': {
      reward += 0.05; // minor positive
      break;
    }

    case 'deploy_honeypot': {
      reward += 0.08;
      break;
    }

    case 'observe_network': {
      // small penalty if observing too much (should be acting)
      if (actionHistory.filter(a => a.startsWith('observe_network')).length > 3) {
        reward -= 0.05;
      }
      break;
    }
  }

  return Math.round(reward * 100) / 100;
}

// final episode grading (5 components)
export function calculateFinalScore(
  phases: AttackPhase[],
  alerts: Alert[],
  networkState: NetworkState,
  stepsUsed: number,
  maxSteps: number,
  actionHistory: string[],
): {
  finalScore: number;
  threatResponse: number;
  falsePositiveHandling: number;
  dataProtection: number;
  collateralDamage: number;
  efficiency: number;
  details: Record<string, unknown>;
} {
  const totalThreats = phases.length;
  const neutralized = phases.filter(p => p.isNeutralized).length;
  const totalFPs = alerts.filter(a => a.isFalsePositive).length;
  const dismissedFPs = alerts.filter(a => a.isFalsePositive && a.status === 'dismissed').length;

  // count healthy hosts that got isolated (collateral)
  const healthyIsolated = networkState.nodes.filter(
    n => n.status === 'isolated' && !phases.some(p => p.targetNodeId === n.nodeId),
  ).length;

  // data exfiltration — check if any exfil phases completed
  const exfilPhases = phases.filter(p => p.compromiseEffect === 'exfiltrated' && !p.isNeutralized);
  const dataExfilMb = exfilPhases.reduce((sum, p) => {
    const progress = p.stepsElapsed / p.stepsToComplete;
    return sum + progress * 500; // up to 500MB per exfil phase
  }, 0);

  // 1. threat response (35%)
  const threatResponse = totalThreats > 0
    ? (neutralized / totalThreats) * 0.35
    : 0.35;

  // 2. false positive handling (20%)
  const falsePositiveHandling = totalFPs > 0
    ? (dismissedFPs / totalFPs) * 0.20
    : 0.20;

  // 3. data protection (20%)
  const dataProtection = dataExfilMb === 0
    ? 0.20
    : Math.max(0, 0.20 * (1 - dataExfilMb / 1000));

  // 4. collateral damage (15%) — less is better
  const maxCollateral = networkState.nodes.filter(n => n.type !== 'honeypot').length;
  const collateralDamage = healthyIsolated === 0
    ? 0.15
    : Math.max(0, 0.15 * (1 - healthyIsolated / maxCollateral));

  // 5. efficiency (10%) — fewer steps is better
  const stepRatio = stepsUsed / maxSteps;
  const efficiency = stepRatio <= 0.6
    ? 0.10
    : Math.max(0, 0.10 * (1 - (stepRatio - 0.6) / 0.4));

  const finalScore = Math.round(
    (threatResponse + falsePositiveHandling + dataProtection + collateralDamage + efficiency) * 1000,
  ) / 1000;

  return {
    finalScore: Math.min(1.0, finalScore),
    threatResponse: Math.round(threatResponse * 1000) / 1000,
    falsePositiveHandling: Math.round(falsePositiveHandling * 1000) / 1000,
    dataProtection: Math.round(dataProtection * 1000) / 1000,
    collateralDamage: Math.round(collateralDamage * 1000) / 1000,
    efficiency: Math.round(efficiency * 1000) / 1000,
    details: {
      threatsNeutralized: neutralized,
      totalThreats,
      fpDismissed: dismissedFPs,
      totalFps: totalFPs,
      stepsUsed,
      maxSteps,
      dataExfiltratedMb: Math.round(dataExfilMb * 10) / 10,
      healthyHostsIsolated: healthyIsolated,
      adversaryBehavior: 'determined at runtime',
    },
  };
}
