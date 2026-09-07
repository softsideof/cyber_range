// web worker entry point for cyber range simulation engine
// runs the attack progression and defender agent off the main thread

import type { WorkerCommand, WorkerMessage } from './types';
import { SCENARIOS } from './scenarios';
import { buildCustomScenario } from './attack-builder';
import { createSimulation, stepSimulation, type SimulationInstance } from './simulation';

let currentSim: SimulationInstance | null = null;
let timerId: ReturnType<typeof setInterval> | null = null;
let intervalMs = 1200; // default 1x speed
let isPaused = false;

function post(msg: WorkerMessage): void {
  self.postMessage(msg);
}

function stopLoop(): void {
  if (timerId !== null) {
    clearInterval(timerId);
    timerId = null;
  }
}

function tick(): void {
  if (!currentSim || isPaused) return;

  const { isDone, state } = stepSimulation(currentSim);
  post({ type: 'STATE_UPDATE', state });

  if (isDone) {
    stopLoop();
    post({ type: 'EPISODE_END', state });
  }
}

function startLoop(): void {
  stopLoop();
  if (isPaused || !currentSim) return;
  timerId = setInterval(tick, intervalMs);
}

self.onmessage = (event: MessageEvent<WorkerCommand>) => {
  const cmd = event.data;

  try {
    switch (cmd.type) {
      case 'LAUNCH': {
        const scenario = SCENARIOS[cmd.scenarioId];
        if (!scenario) {
          post({ type: 'ERROR', message: `Scenario ${cmd.scenarioId} not found.` });
          return;
        }
        currentSim = createSimulation(scenario);
        isPaused = false;
        // immediately send initial state
        const { state } = stepSimulation(currentSim);
        post({ type: 'STATE_UPDATE', state });
        startLoop();
        break;
      }

      case 'LAUNCH_CUSTOM': {
        const scenario = buildCustomScenario(cmd.config);
        currentSim = createSimulation(scenario);
        isPaused = false;
        const { state } = stepSimulation(currentSim);
        post({ type: 'STATE_UPDATE', state });
        startLoop();
        break;
      }

      case 'PAUSE': {
        isPaused = true;
        stopLoop();
        break;
      }

      case 'RESUME': {
        isPaused = false;
        startLoop();
        break;
      }

      case 'SET_SPEED': {
        // speed multiplier: 1x -> 1200ms, 2x -> 600ms, 4x -> 300ms
        const baseMs = 1200;
        intervalMs = Math.max(150, Math.round(baseMs / (cmd.speed || 1)));
        if (!isPaused && currentSim) {
          startLoop();
        }
        break;
      }

      case 'STEP_ONCE': {
        if (!currentSim) return;
        isPaused = true;
        stopLoop();
        const { isDone, state } = stepSimulation(currentSim);
        post({ type: 'STATE_UPDATE', state });
        if (isDone) {
          post({ type: 'EPISODE_END', state });
        }
        break;
      }

      case 'RESET': {
        if (!currentSim) return;
        stopLoop();
        const scenario = currentSim.scenario;
        currentSim = createSimulation(scenario);
        isPaused = true;
        const { state } = stepSimulation(currentSim);
        post({ type: 'STATE_UPDATE', state });
        break;
      }

      case 'STOP': {
        stopLoop();
        currentSim = null;
        isPaused = false;
        break;
      }
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    post({ type: 'ERROR', message });
  }
};
