// test incident post-mortem reporter generator
import { describe, it, expect } from 'vitest';
import { generateIncidentReport } from '../../src/engine/reporter';
import { createNetworkState } from '../../src/engine/network';
import type { GraderResult, ScenarioConfig } from '../../src/engine/types';

describe('Incident Post-Mortem Reporter', () => {
  const dummyGrader: GraderResult = {
    finalScore: 0.88,
    threatResponse: 0.35,
    falsePositiveHandling: 0.20,
    dataProtection: 0.18,
    collateralDamage: 0.15,
    efficiency: 0.08,
    details: {
      threatsNeutralized: 3,
      totalThreats: 3,
      fpDismissed: 2,
      totalFps: 2,
      stepsUsed: 8,
      maxSteps: 20,
      dataExfiltratedMb: 12,
      healthyHostsIsolated: 0,
      adversaryBehavior: 'evasive',
    },
    judgeVerdicts: {
      junior: { score: 0.9, verdict: 'All alerts investigated thoroughly.' },
      senior: { score: 0.85, verdict: 'Clean severity triage.' },
      commander: { score: 0.92, verdict: 'Minimal blast radius and data safe.' },
    },
  };

  const dummyScenario: Partial<ScenarioConfig> = {
    id: 'apt_lateral_movement',
    name: 'APT29 Lateral Movement',
    difficulty: 'hard',
  };

  it('generates markdown post-mortem report containing key incident metrics', () => {
    const net = createNetworkState();
    const markdown = generateIncidentReport(
      dummyGrader,
      dummyScenario as ScenarioConfig,
      [
        {
          step: 1,
          action: { tool: 'investigate_alert', args: { alert_id: 'alt-01' }, reasoning: 'test' },
          result: { success: true, description: 'Investigated alert', cost: 2, reward: 0.2, details: {} },
          reward: 0.2,
          phase: 'investigate',
          topologySnapshot: [],
        },
      ],
      net.nodes,
    );

    expect(markdown).toContain('Security Incident Post-Mortem & After-Action Report');
    expect(markdown).toContain('APT29 Lateral Movement');
    expect(markdown).toContain('88%');
    expect(markdown).toContain('Junior SOC Analyst');
    expect(markdown).toContain('Senior Incident Lead');
    expect(markdown).toContain('Incident Commander');
    expect(markdown).toContain('investigate_alert');
  });
});
