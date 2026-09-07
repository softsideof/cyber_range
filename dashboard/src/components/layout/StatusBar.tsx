'use client';

// bottom status bar showing worker state, tick metrics, and keyboard shortcuts

import React from 'react';
import { useAppStore } from '@/store';
import styles from './StatusBar.module.css';

export function StatusBar() {
  const isPaused = useAppStore((s) => s.isPaused);
  const mode = useAppStore((s) => s.mode);
  const step = useAppStore((s) => s.step);
  const maxSteps = useAppStore((s) => s.maxSteps);

  return (
    <footer className={styles.statusBar}>
      <div className={styles.left}>
        <span>
          <span className={styles.indicator} style={{ background: isPaused ? 'var(--yellow)' : 'var(--green)' }} />
          STATUS: {isPaused ? 'PAUSED' : mode.toUpperCase()}
        </span>
        <span>•</span>
        <span>ENGINE: WEB WORKER (CLIENT)</span>
        <span>•</span>
        <span>STEP {step}/{maxSteps}</span>
      </div>

      <div className={styles.right}>
        <span className={styles.keyHint}>
          SHORTCUTS: <kbd>SPACE</kbd> PAUSE &nbsp;|&nbsp; <kbd>B</kbd> VIRUS BUILDER &nbsp;|&nbsp; <kbd>A</kbd> ARCH &nbsp;|&nbsp; <kbd>1-6</kbd> SCENARIOS
        </span>
        <span>•</span>
        <span>CYBERRANGE v0.1 • OPENENV 0.2.2</span>
      </div>
    </footer>
  );
}
