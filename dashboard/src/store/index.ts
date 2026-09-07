// zustand store for cyber range dashboard
// manages simulation data, ui view states, and attack builder modal

import { create } from 'zustand';
import type {
  ScenarioConfig,
  NetworkNode,
  Alert,
  AgentLogEntry,
  SimMetrics,
  MitreEntry,
  GraderResult,
  SocPhase,
  SimulationState,
} from '../engine/types';
import { createDefaultTopology } from '../engine/network';
import { SCENARIOS } from '../engine/scenarios';

export interface AppStore {
  // simulation slice
  mode: SimulationState['mode'];
  scenario: ScenarioConfig | null;
  step: number;
  maxSteps: number;
  topology: NetworkNode[];
  alerts: Alert[];
  agentLog: AgentLogEntry[];
  metrics: SimMetrics;
  mitre: MitreEntry[];
  graderResult: GraderResult | null;
  currentPhase: SocPhase;

  // ui slice
  speed: 1 | 2 | 4;
  isPaused: boolean;
  isMobile: boolean;
  activeTab: 'network' | 'alerts' | 'agent' | 'mitre' | 'intel' | 'score';
  showAttackBuilder: boolean;
  showArchitecture: boolean;
  showBriefing: boolean;
  showShortcuts: boolean;
  selectedNodeId: string | null;
  viewMode: '3d' | '2d';
  autoDemo: boolean;
  currentScenarioIndex: number;

  // actions
  setSimulationState: (state: SimulationState) => void;
  setPaused: (isPaused: boolean) => void;
  togglePaused: () => void;
  setSpeed: (speed: 1 | 2 | 4) => void;
  cycleSpeed: () => void;
  setActiveTab: (tab: 'network' | 'alerts' | 'agent' | 'mitre' | 'intel' | 'score') => void;
  setShowAttackBuilder: (show: boolean) => void;
  setShowArchitecture: (show: boolean) => void;
  setShowBriefing: (show: boolean) => void;
  setShowShortcuts: (show: boolean) => void;
  setSelectedNodeId: (nodeId: string | null) => void;
  setViewMode: (mode: '3d' | '2d') => void;
  setAutoDemo: (auto: boolean) => void;
  nextScenario: () => ScenarioConfig;
  loadScenario: (scenarioId: string) => ScenarioConfig | null;
}

const DEFAULT_METRICS: SimMetrics = {
  health: 100,
  threatLevel: 'green',
  budget: 100,
  maxBudget: 100,
  dataAtRiskMb: 0,
  score: 0,
};

const SCENARIO_KEYS = Object.keys(SCENARIOS);

export const useAppStore = create<AppStore>((set, get) => ({
  // initial simulation state
  mode: 'idle',
  scenario: SCENARIOS['script_kiddie'] || null,
  step: 0,
  maxSteps: 25,
  topology: createDefaultTopology(),
  alerts: [],
  agentLog: [],
  metrics: DEFAULT_METRICS,
  mitre: [],
  graderResult: null,
  currentPhase: 'triage',

  // initial ui state
  speed: 1,
  isPaused: true,
  isMobile: false,
  activeTab: 'network',
  showAttackBuilder: false,
  showArchitecture: false,
  showBriefing: false,
  showShortcuts: false,
  selectedNodeId: null,
  viewMode: '2d',
  autoDemo: false,
  currentScenarioIndex: 0,

  setSimulationState: (state) => {
    set({
      mode: state.mode,
      scenario: state.scenario,
      step: state.step,
      maxSteps: state.maxSteps,
      topology: state.topology,
      alerts: state.alerts,
      agentLog: state.agentLog,
      metrics: state.metrics,
      mitre: state.mitre,
      graderResult: state.graderResult,
      currentPhase: state.currentPhase,
    });
  },

  setPaused: (isPaused) => set({ isPaused }),
  togglePaused: () => set((state) => ({ isPaused: !state.isPaused })),

  setSpeed: (speed) => set({ speed }),
  cycleSpeed: () => {
    const speeds: Array<1 | 2 | 4> = [1, 2, 4];
    const curr = get().speed;
    const next = speeds[(speeds.indexOf(curr) + 1) % speeds.length];
    set({ speed: next });
  },

  setActiveTab: (activeTab) => set({ activeTab }),
  setShowAttackBuilder: (showAttackBuilder) => set({ showAttackBuilder }),
  setShowArchitecture: (showArchitecture) => set({ showArchitecture }),
  setShowBriefing: (showBriefing) => set({ showBriefing }),
  setShowShortcuts: (showShortcuts) => set({ showShortcuts }),
  setSelectedNodeId: (selectedNodeId) => set({ selectedNodeId }),
  setViewMode: (viewMode) => set({ viewMode }),
  setAutoDemo: (autoDemo) => set({ autoDemo }),

  nextScenario: () => {
    const nextIdx = (get().currentScenarioIndex + 1) % SCENARIO_KEYS.length;
    const scenarioKey = SCENARIO_KEYS[nextIdx];
    const scenario = SCENARIOS[scenarioKey];
    set({
      currentScenarioIndex: nextIdx,
      scenario,
      step: 0,
      alerts: [],
      agentLog: [],
      graderResult: null,
      topology: createDefaultTopology(),
    });
    return scenario;
  },

  loadScenario: (scenarioId) => {
    const scenario = SCENARIOS[scenarioId];
    if (scenario) {
      const idx = SCENARIO_KEYS.indexOf(scenarioId);
      set({
        currentScenarioIndex: idx >= 0 ? idx : 0,
        scenario,
        step: 0,
        alerts: [],
        agentLog: [],
        graderResult: null,
        topology: createDefaultTopology(),
      });
      return scenario;
    }
    return null;
  },
}));
