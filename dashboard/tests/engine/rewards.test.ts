// simulation engine unit tests
import { describe, it, expect } from 'vitest';
import { calculateStepReward, calculateFinalScore } from '../../src/engine/rewards';
import { createNetworkState } from '../../src/engine/network';
import type { Alert, AttackPhase, AgentAction } from '../../src/engine/types';

describe('reward calculator', () => {
  it('rewards investigating high severity alerts', () => {
    const netState = createNetworkState();
    const action: AgentAction = {
      tool: 'investigate_alert',
      args: { alert_id: 'ALT-1' },
      reasoning: 'investigating critical alert',
    };
    const alerts: Alert[] = [
      {
        alertId: 'ALT-1',
        severity: 'critical',
        type: 'intrusion',
        title: 'Exploit detected',
        sourceIp: '198.51.100.1',
        targetNodeId: 'web-01',
        description: 'test',
        timestamp: 1,
        status: 'new',
        isFalsePositive: false,
        forensicEvidence: 'malicious',
        mitreId: 'T1190',
        confidence: 0.9,
      },
    ];

    const reward = calculateStepReward(action, netState, alerts, [], []);
    expect(reward).toBeGreaterThan(0.1);
  });

  it('penalizes isolating healthy host collateral damage', () => {
    const netState = createNetworkState();
    const action: AgentAction = {
      tool: 'isolate_host',
      args: { node_id: 'ws-01' },
      reasoning: 'isolating host',
    };

    const reward = calculateStepReward(action, netState, [], [], []);
    expect(reward).toBeLessThan(0);
  });

  it('calculates bounded 5-component final score', () => {
    const netState = createNetworkState();
    const phases: AttackPhase[] = [
      {
        phaseId: 'P1',
        name: 'Infiltration',
        description: 'test',
        targetNodeId: 'ws-01',
        attackType: 'malware',
        stepsToComplete: 5,
        stepsElapsed: 2,
        isActive: false,
        isNeutralized: true,
        mitreId: 'T1059',
        mitreName: 'Scripting',
        mitreTactic: 'Execution',
        compromiseEffect: 'compromised',
        sourceIp: '198.51.100.2',
      },
    ];

    const result = calculateFinalScore(phases, [], netState, 10, 25, []);
    expect(result.finalScore).toBeGreaterThan(0);
    expect(result.finalScore).toBeLessThanOrEqual(1.0);
    expect(result.threatResponse).toBeGreaterThan(0);
  });
});
