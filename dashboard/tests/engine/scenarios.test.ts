// scenario catalog and configuration unit tests
import { describe, it, expect } from 'vitest';
import { SCENARIO_ORDER, getScenario } from '../../src/engine/scenarios';

describe('Attack Scenarios Catalog', () => {
  it('contains all 6 core attack scenarios', () => {
    expect(SCENARIO_ORDER.length).toBe(6);
    expect(SCENARIO_ORDER).toContain('ransomware_outbreak');
    expect(SCENARIO_ORDER).toContain('apt_lateral_movement');
    expect(SCENARIO_ORDER).toContain('supply_chain_compromise');
  });

  it('validates scenario structures and alert payloads', () => {
    for (const id of SCENARIO_ORDER) {
      const scenario = getScenario(id);
      expect(scenario).toBeDefined();
      if (!scenario) return;

      expect(scenario.name).toBeTruthy();
      expect(scenario.attackPhases.length).toBeGreaterThan(0);
      expect(scenario.alerts.length).toBeGreaterThan(0);
      expect(scenario.maxSteps).toBeGreaterThan(5);

      // check that false positives exist for triage testing
      const fps = scenario.alerts.filter((a) => a.isFalsePositive);
      const threats = scenario.alerts.filter((a) => !a.isFalsePositive);
      expect(fps.length).toBeGreaterThan(0);
      expect(threats.length).toBeGreaterThan(0);

      // check MITRE tactic mappings and attack properties
      scenario.attackPhases.forEach((p) => {
        expect(p.mitreTactic).toBeTruthy();
        expect(p.targetNodeId).toBeTruthy();
        expect(p.compromiseEffect).toBeTruthy();
      });
    }
  });

  it('returns undefined for non-existent scenario query', () => {
    const fallback = getScenario('non_existent_scenario_xyz');
    expect(fallback).toBeUndefined();
  });
});
