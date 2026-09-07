// simulation orchestrator
// ties together network topology, adversary progression, defender agent, rewards, and grading

import type {
  ScenarioConfig,
  SimulationState,
  AgentLogEntry,
  SimMetrics,
  MitreEntry,
  Alert,
  AttackPhase,
  GraderResult,
} from './types';
import {
  createNetworkState,
  getNetworkHealth,
  getThreatLevel,
  applyAction,
  type NetworkState,
} from './network';
import { progressAttacks, countActiveThreats } from './attack-runner';
import { createDefenderState, decide } from './defender';
import { calculateStepReward } from './rewards';
import { gradeEpisode } from './grader';

export interface SimulationInstance {
  scenario: ScenarioConfig;
  networkState: NetworkState;
  defender: ReturnType<typeof createDefenderState>;
  phases: AttackPhase[];
  allAlerts: Alert[];      // combined queue from scenario + runtime
  activeAlerts: Alert[];   // alerts revealed up to current step
  actionHistory: string[];
  agentLog: AgentLogEntry[];
  step: number;
  maxSteps: number;
  budget: number;
  maxBudget: number;
  runningScore: number;
  mode: SimulationState['mode'];
  graderResult: GraderResult | null;
}

// initialize a fresh simulation from scenario config
export function createSimulation(scenario: ScenarioConfig): SimulationInstance {
  const networkState = createNetworkState(scenario.initialCompromisedNodes || []);
  const defender = createDefenderState();
  const phases = structuredClone(scenario.attackPhases);

  // combine threat alerts and false positive alerts
  const combinedAlerts: Alert[] = [
    ...scenario.alerts,
    ...(scenario.falsePositiveAlerts || []),
  ].sort((a, b) => a.timestamp - b.timestamp);

  // budget scaled with max steps
  const maxBudget = scenario.maxSteps * 4;

  return {
    scenario,
    networkState,
    defender,
    phases,
    allAlerts: combinedAlerts,
    activeAlerts: [],
    actionHistory: [],
    agentLog: [],
    step: 0,
    maxSteps: scenario.maxSteps,
    budget: maxBudget,
    maxBudget,
    runningScore: 0,
    mode: 'running',
    graderResult: null,
  };
}

// execute one step in the simulation cycle
export function stepSimulation(sim: SimulationInstance): {
  isDone: boolean;
  state: SimulationState;
} {
  if (sim.mode === 'complete' || sim.step >= sim.maxSteps) {
    sim.mode = 'complete';
    return { isDone: true, state: getSimulationState(sim) };
  }

  sim.step++;

  // 1. reveal alerts scheduled for this step or earlier
  const newIncoming = sim.allAlerts.filter(
    a => a.timestamp <= sim.step && !sim.activeAlerts.some(curr => curr.alertId === a.alertId),
  );
  if (newIncoming.length > 0) {
    sim.activeAlerts.push(...newIncoming);
  }

  // 2. activate attack phases sequentially as time progresses
  const activeCount = countActiveThreats(sim.phases);
  if (activeCount === 0) {
    const nextInactive = sim.phases.find(p => !p.isActive && !p.isNeutralized && p.stepsElapsed === 0);
    if (nextInactive) {
      nextInactive.isActive = true;
    }
  }

  // 3. progress adversary attacks and adversary evasions
  const { newAlerts: runtimeAlerts } = progressAttacks(
    sim.phases,
    sim.networkState,
    sim.scenario.adversaryBehavior,
    sim.step,
  );
  if (runtimeAlerts.length > 0) {
    sim.activeAlerts.push(...runtimeAlerts);
  }

  // 4. defender AI agent observes and takes an action
  const action = decide(sim.defender, sim.activeAlerts, sim.networkState, sim.scenario.difficulty);

  // 5. apply action to network state
  const actionRes = applyAction(sim.networkState, action.tool, action.args);
  sim.budget = Math.max(0, sim.budget - actionRes.cost);

  // 6. update alert status when investigated or dismissed
  if (action.tool === 'investigate_alert' && action.args.alert_id) {
    const targetAlert = sim.activeAlerts.find(a => a.alertId === action.args.alert_id);
    if (targetAlert) targetAlert.status = 'investigating';
  } else if (action.tool === 'dismiss_alert' && action.args.alert_id) {
    const targetAlert = sim.activeAlerts.find(a => a.alertId === action.args.alert_id);
    if (targetAlert) targetAlert.status = 'dismissed';
  } else if (action.tool === 'isolate_host' && action.args.node_id) {
    sim.activeAlerts
      .filter(a => a.targetNodeId === action.args.node_id)
      .forEach(a => { a.status = 'contained'; });
  }

  // 7. calculate step reward
  const stepReward = calculateStepReward(
    action,
    sim.networkState,
    sim.activeAlerts,
    sim.phases,
    sim.actionHistory,
  );
  sim.runningScore += stepReward;

  const actionKey = `${action.tool}:${JSON.stringify(action.args)}`;
  sim.actionHistory.push(actionKey);

  // record agent log entry
  const logEntry: AgentLogEntry = {
    step: sim.step,
    action,
    result: {
      ...actionRes,
      reward: stepReward,
      details: {},
    },
    reward: stepReward,
    phase: sim.defender.currentPhase,
    topologySnapshot: sim.networkState.nodes.map(n => n.status),
  };
  sim.agentLog.push(logEntry);

  // 8. check terminal conditions
  const allNeutralized = sim.phases.every(p => p.isNeutralized || (!p.isActive && p.stepsElapsed > 0));
  const maxStepsReached = sim.step >= sim.maxSteps;
  const budgetDepleted = sim.budget <= 0;

  const isDone = allNeutralized || maxStepsReached || budgetDepleted;

  if (isDone) {
    sim.mode = 'complete';
    sim.graderResult = gradeEpisode(
      sim.phases,
      sim.activeAlerts,
      sim.networkState,
      sim.step,
      sim.maxSteps,
      sim.actionHistory,
      sim.scenario.adversaryBehavior,
    );
  }

  return { isDone, state: getSimulationState(sim) };
}

// extract snapshot for main thread or UI rendering
export function getSimulationState(sim: SimulationInstance): SimulationState {
  const health = getNetworkHealth(sim.networkState);
  const threatLevel = getThreatLevel(sim.networkState);

  // estimate data at risk based on active exfil phases
  const dataAtRiskMb = sim.phases
    .filter(p => p.compromiseEffect === 'exfiltrated' && !p.isNeutralized)
    .reduce((sum, p) => sum + (1 - p.stepsElapsed / p.stepsToComplete) * 500, 0);

  const metrics: SimMetrics = {
    health,
    threatLevel,
    budget: sim.budget,
    maxBudget: sim.maxBudget,
    dataAtRiskMb: Math.max(0, Math.round(dataAtRiskMb)),
    score: Math.round(sim.runningScore * 100) / 100,
  };

  // generate MITRE technique matrix states
  const mitre: MitreEntry[] = sim.phases.map(phase => {
    let state: MitreEntry['state'] = 'inactive';
    if (phase.isNeutralized) {
      state = 'defended';
    } else if (phase.isActive || phase.stepsElapsed > 0) {
      state = 'attacking';
    }
    return {
      id: phase.mitreId,
      name: phase.mitreName || phase.name,
      tactic: phase.mitreTactic || 'Execution',
      state,
    };
  });

  return {
    mode: sim.mode,
    scenario: sim.scenario,
    step: sim.step,
    maxSteps: sim.maxSteps,
    topology: structuredClone(sim.networkState.nodes),
    alerts: structuredClone(sim.activeAlerts),
    agentLog: structuredClone(sim.agentLog),
    metrics,
    mitre,
    graderResult: sim.graderResult,
    currentPhase: sim.defender.currentPhase,
  };
}
