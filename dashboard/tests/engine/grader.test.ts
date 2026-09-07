// grader and cyberjudge unit tests
import { describe, it, expect } from 'vitest';
import { gradeEpisode } from '../../src/engine/grader';
import { createNetworkState } from '../../src/engine/network';
import type { AttackPhase, Alert } from '../../src/engine/types';

describe('CyberJudge Grader', () => {
  const samplePhases: AttackPhase[] = [
    {
      id: 'p1',
      name: 'Initial Access',
      description: 'Phishing payload execution',
      targetNodeId: 'ws-01',
      actionRequired: 'isolate_host',
      completed: true,
      detectionChance: 0.8,
      noiseLevel: 0.3,
      mitreTactic: 'initial-access',
      mitreTechnique: 'T1566',
    },
  ];

  const sampleAlerts: Alert[] = [
    {
      alertId: 'alt-01',
      severity: 'critical',
      type: 'malware',
      title: 'Payload Detected',
      sourceIp: '198.51.100.24',
      targetNodeId: 'ws-01',
      description: 'C2 payload execution',
      timestamp: 1,
      status: 'resolved',
      isFalsePositive: false,
      forensicEvidence: 'SHA-256 matched trojan binary',
      mitreId: 'T1566',
      confidence: 0.95,
    },
    {
      alertId: 'alt-02',
      severity: 'low',
      type: 'recon',
      title: 'Port Scan',
      sourceIp: '10.0.0.99',
      targetNodeId: 'fw-01',
      description: 'Routine probe',
      timestamp: 2,
      status: 'dismissed',
      isFalsePositive: true,
      forensicEvidence: 'Approved internal audit scanner',
      mitreId: 'T1046',
      confidence: 0.4,
    },
  ];

  it('produces valid scores and verdicts for all three judge personas', () => {
    const net = createNetworkState();
    const result = gradeEpisode(
      samplePhases,
      sampleAlerts,
      net,
      5,
      20,
      ['investigate_alert:alt-01', 'isolate_host:ws-01', 'dismiss_alert:alt-02'],
      'static',
    );

    expect(result.finalScore).toBeGreaterThanOrEqual(0);
    expect(result.finalScore).toBeLessThanOrEqual(1);

    // ensure all 3 judge personas have scores and human-readable feedback
    expect(result.judgeVerdicts.junior.score).toBeGreaterThanOrEqual(0);
    expect(result.judgeVerdicts.junior.verdict).toBeTruthy();

    expect(result.judgeVerdicts.senior.score).toBeGreaterThanOrEqual(0);
    expect(result.judgeVerdicts.senior.verdict).toBeTruthy();

    expect(result.judgeVerdicts.commander.score).toBeGreaterThanOrEqual(0);
    expect(result.judgeVerdicts.commander.verdict).toBeTruthy();
  });

  it('accurately reports detail metrics for post-incident review', () => {
    const net = createNetworkState();
    const result = gradeEpisode(samplePhases, sampleAlerts, net, 3, 15, [], 'evasive');

    expect(result.details.totalThreats).toBe(1);
    expect(result.details.totalFps).toBe(1);
    expect(result.details.adversaryBehavior).toBe('evasive');
    expect(result.details.stepsUsed).toBe(3);
    expect(result.details.maxSteps).toBe(15);
  });
});
