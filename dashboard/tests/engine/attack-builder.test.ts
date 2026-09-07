// simulation engine unit tests
import { describe, it, expect } from 'vitest';
import { buildCustomScenario, VIRUS_TEMPLATES } from '../../src/engine/attack-builder';

describe('attack builder & virus templates', () => {
  it('builds valid scenario from WannaCry Lite template', () => {
    const template = VIRUS_TEMPLATES['wannacry_lite'];
    const scenario = buildCustomScenario(template);

    expect(scenario.name).toContain('WannaCry');
    expect(scenario.attackPhases.length).toBe(4);
    expect(scenario.adversaryBehavior).toBe('persistent');
    expect(scenario.mitreTechniques).toContain('T1486');
    expect(scenario.falsePositiveAlerts.length).toBeGreaterThan(0);
  });

  it('builds valid scenario from SolarWinds Jr template', () => {
    const template = VIRUS_TEMPLATES['solarwinds_jr'];
    const scenario = buildCustomScenario(template);

    expect(scenario.name).toContain('SolarWinds');
    expect(scenario.adversaryBehavior).toBe('adaptive');
    expect(scenario.attackPhases.some((p) => p.targetNodeId === 'dc-01')).toBe(true);
  });

  it('builds valid bespoke custom attack', () => {
    const scenario = buildCustomScenario({
      name: 'Test APT Infiltration',
      description: 'Test description',
      attackVector: 'ransomware',
      targets: ['ws-04', 'backup-01'],
      evasion: {
        rotateC2: false,
        recompromise: false,
        decoyAlerts: false,
      },
      difficulty: 'easy',
      maxSteps: 20,
      phases: [
        { name: 'Initial Ransomware Drop', target: 'ws-04', type: 'ransomware', mitreId: 'T1486' },
      ],
    });

    expect(scenario.adversaryBehavior).toBe('static');
    expect(scenario.attackPhases.length).toBe(1);
    expect(scenario.alerts.length).toBe(1);
    expect(scenario.alerts[0].forensicEvidence).toContain('reverse shell');
  });
});
