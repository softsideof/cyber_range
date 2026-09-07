// hook for managing the simulation runner
// supports both Web Worker execution and direct fallback runner for maximum compatibility

import { useEffect, useRef, useCallback } from 'react';
import { useAppStore } from '../store';
import type { CustomAttackConfig, ScenarioConfig, SimulationState, WorkerCommand, WorkerMessage } from '../engine/types';
import { SCENARIOS } from '../engine/scenarios';
import { buildCustomScenario } from '../engine/attack-builder';
import { createSimulation, stepSimulation, type SimulationInstance } from '../engine/simulation';

export function useWorker() {
  const workerRef = useRef<Worker | null>(null);
  const directSimRef = useRef<SimulationInstance | null>(null);
  const directTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const isPaused = useAppStore(s => s.isPaused);
  const speed = useAppStore(s => s.speed);
  const setSimulationState = useAppStore(s => s.setSimulationState);
  const setPaused = useAppStore(s => s.setPaused);
  const autoDemo = useAppStore(s => s.autoDemo);
  const nextScenario = useAppStore(s => s.nextScenario);

  // interval ms based on speed (1x = 1200ms, 2x = 600ms, 4x = 300ms)
  const getIntervalMs = useCallback((s: number) => {
    return Math.max(150, Math.round(1200 / s));
  }, []);

  // fallback direct runner tick
  const directTick = useCallback(() => {
    if (!directSimRef.current || useAppStore.getState().isPaused) return;

    const { isDone, state } = stepSimulation(directSimRef.current);
    setSimulationState(state);

    if (isDone) {
      if (directTimerRef.current) {
        clearInterval(directTimerRef.current);
        directTimerRef.current = null;
      }
      // auto-demo scenario progression
      if (useAppStore.getState().autoDemo) {
        setTimeout(() => {
          if (useAppStore.getState().autoDemo) {
            const next = nextScenario();
            startDirectSim(next);
          }
        }, 5000);
      }
    }
  }, [setSimulationState, autoDemo, nextScenario]);

  const startDirectSim = useCallback((scenario: ScenarioConfig) => {
    if (directTimerRef.current) {
      clearInterval(directTimerRef.current);
      directTimerRef.current = null;
    }
    directSimRef.current = createSimulation(scenario);
    setPaused(false);
    const { state } = stepSimulation(directSimRef.current);
    setSimulationState(state);
    directTimerRef.current = setInterval(directTick, getIntervalMs(useAppStore.getState().speed));
  }, [directTick, getIntervalMs, setPaused, setSimulationState]);

  // initialize worker on mount
  useEffect(() => {
    let workerSupported = false;

    try {
      if (typeof window !== 'undefined' && window.Worker) {
        const worker = new Worker(new URL('../engine/worker.ts', import.meta.url), {
          type: 'module',
        });

        worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
          const msg = event.data;
          if (msg.type === 'STATE_UPDATE' || msg.type === 'EPISODE_END') {
            setSimulationState(msg.state);

            if (msg.type === 'EPISODE_END' && useAppStore.getState().autoDemo) {
              setTimeout(() => {
                if (useAppStore.getState().autoDemo) {
                  const next = nextScenario();
                  worker.postMessage({ type: 'LAUNCH', scenarioId: next.id } satisfies WorkerCommand);
                }
              }, 5000);
            }
          } else if (msg.type === 'ERROR') {
            console.warn('Worker reported error, falling back to direct runner:', msg.message);
          }
        };

        workerRef.current = worker;
        workerSupported = true;
      }
    } catch {
      // web worker module loading can be restricted in some environments; direct fallback handles it
      workerSupported = false;
    }

    // start initial scenario
    const initialScenario = useAppStore.getState().scenario || SCENARIOS['script_kiddie'];
    if (workerSupported && workerRef.current) {
      workerRef.current.postMessage({ type: 'LAUNCH', scenarioId: initialScenario.id } satisfies WorkerCommand);
    } else {
      startDirectSim(initialScenario);
    }

    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
      if (directTimerRef.current) {
        clearInterval(directTimerRef.current);
        directTimerRef.current = null;
      }
    };
  }, [startDirectSim, setSimulationState, nextScenario]);

  // respond to pause / resume
  useEffect(() => {
    if (workerRef.current) {
      workerRef.current.postMessage({ type: isPaused ? 'PAUSE' : 'RESUME' } satisfies WorkerCommand);
    } else {
      if (isPaused && directTimerRef.current) {
        clearInterval(directTimerRef.current);
        directTimerRef.current = null;
      } else if (!isPaused && directSimRef.current && !directTimerRef.current) {
        directTimerRef.current = setInterval(directTick, getIntervalMs(speed));
      }
    }
  }, [isPaused, directTick, getIntervalMs, speed]);

  // respond to speed changes
  useEffect(() => {
    if (workerRef.current) {
      workerRef.current.postMessage({ type: 'SET_SPEED', speed } satisfies WorkerCommand);
    } else {
      if (directTimerRef.current && !isPaused) {
        clearInterval(directTimerRef.current);
        directTimerRef.current = setInterval(directTick, getIntervalMs(speed));
      }
    }
  }, [speed, isPaused, directTick, getIntervalMs]);

  const launch = useCallback((scenarioId: string) => {
    useAppStore.getState().setAutoDemo(false);
    const scenario = useAppStore.getState().loadScenario(scenarioId);
    if (!scenario) return;

    if (workerRef.current) {
      workerRef.current.postMessage({ type: 'LAUNCH', scenarioId } satisfies WorkerCommand);
    } else {
      startDirectSim(scenario);
    }
  }, [startDirectSim]);

  const launchCustom = useCallback((config: CustomAttackConfig) => {
    useAppStore.getState().setAutoDemo(false);
    const scenario = buildCustomScenario(config);
    useAppStore.setState({
      scenario,
      step: 0,
      alerts: [],
      agentLog: [],
      graderResult: null,
    });

    if (workerRef.current) {
      workerRef.current.postMessage({ type: 'LAUNCH_CUSTOM', config } satisfies WorkerCommand);
    } else {
      startDirectSim(scenario);
    }
  }, [startDirectSim]);

  const restart = useCallback(() => {
    const current = useAppStore.getState().scenario || SCENARIOS['script_kiddie'];
    useAppStore.getState().setPaused(true);
    if (directTimerRef.current) {
      clearInterval(directTimerRef.current);
      directTimerRef.current = null;
    }
    directSimRef.current = createSimulation(current);
    const { state } = stepSimulation(directSimRef.current);
    setSimulationState(state);
  }, [setSimulationState]);

  const stepOnce = useCallback(() => {
    if (!directSimRef.current) {
      const current = useAppStore.getState().scenario || SCENARIOS['script_kiddie'];
      directSimRef.current = createSimulation(current);
    }
    const { isDone, state } = stepSimulation(directSimRef.current);
    setSimulationState(state);
    if (isDone && directTimerRef.current) {
      clearInterval(directTimerRef.current);
      directTimerRef.current = null;
    }
  }, [setSimulationState]);

  return {
    launch,
    launchCustom,
    restart,
    stepOnce,
  };
}
