// simulation engine unit tests
import { describe, it, expect } from 'vitest';
import { createSimulation, stepSimulation } from '../../src/engine/simulation';
import { SCENARIOS } from '../../src/engine/scenarios';
import { getNetworkHealth } from '../../src/engine/network';

describe('simulation engine', () => {
  it('initializes clean network and correct step count', () => {
    const scenario = SCENARIOS['script_kiddie'];
    const sim = createSimulation(scenario);

    expect(sim.step).toBe(0);
    expect(sim.maxSteps).toBe(scenario.maxSteps);
    expect(sim.networkState.nodes.length).toBe(12);
    expect(getNetworkHealth(sim.networkState)).toBe(100);
  });

  it('advances simulation state and generates agent log', () => {
    const scenario = SCENARIOS['script_kiddie'];
    const sim = createSimulation(scenario);

    const { state, isDone } = stepSimulation(sim);
    expect(state.step).toBe(1);
    expect(state.agentLog.length).toBe(1);
    expect(state.agentLog[0].action.tool).toBe('observe_network');
    expect(isDone).toBe(false);
  });

  it('reaches completion and produces grader result', () => {
    const scenario = SCENARIOS['script_kiddie'];
    const sim = createSimulation(scenario);

    // run until completion
    let done = false;
    let finalState = null;
    while (!done) {
      const res = stepSimulation(sim);
      done = res.isDone;
      finalState = res.state;
    }

    expect(done).toBe(true);
    expect(finalState?.graderResult).toBeDefined();
    expect(finalState?.graderResult?.finalScore).toBeGreaterThanOrEqual(0);
    expect(finalState?.graderResult?.finalScore).toBeLessThanOrEqual(1.0);
    expect(finalState?.graderResult?.judgeVerdicts.junior).toBeDefined();
    expect(finalState?.graderResult?.judgeVerdicts.senior).toBeDefined();
    expect(finalState?.graderResult?.judgeVerdicts.commander).toBeDefined();
  });
});
