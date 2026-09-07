// keyboard shortcuts for operational power-users
// Space = pause/play, B = attack builder, A = architecture, 1-6 = scenarios, Esc = close modals

import { useEffect } from 'react';
import { useAppStore } from '../store';
import { SCENARIOS } from '../engine/scenarios';

const SCENARIO_KEYS = Object.keys(SCENARIOS);

export function useKeyboard(launchScenario: (id: string) => void) {
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // ignore when focusing inputs, textareas, or select dropdowns
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)
      ) {
        return;
      }

      const store = useAppStore.getState();

      if (e.code === 'Space') {
        e.preventDefault();
        store.togglePaused();
      } else if (e.key === 'b' || e.key === 'B') {
        e.preventDefault();
        store.setShowAttackBuilder(!store.showAttackBuilder);
      } else if (e.key === 'a' || e.key === 'A') {
        e.preventDefault();
        store.setShowArchitecture(!store.showArchitecture);
      } else if (e.key === 'Escape') {
        e.preventDefault();
        store.setShowAttackBuilder(false);
        store.setShowArchitecture(false);
        store.setShowBriefing(false);
        store.setSelectedNodeId(null);
      } else if (['1', '2', '3', '4', '5', '6'].includes(e.key)) {
        const num = parseInt(e.key, 10) - 1;
        if (num >= 0 && num < SCENARIO_KEYS.length) {
          e.preventDefault();
          launchScenario(SCENARIO_KEYS[num]);
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [launchScenario]);
}
