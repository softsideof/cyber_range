'use client';

// accessible keyboard navigation cheatsheet modal
import React from 'react';
import { useAppStore } from '@/store';
import styles from './ShortcutsModal.module.css';

const SHORTCUTS = [
  { key: 'Space', desc: 'Toggle Play / Pause simulation' },
  { key: '→ or .', desc: 'Step forward 1 tactical decision' },
  { key: 'R', desc: 'Reset episode back to Step 0' },
  { key: 'B', desc: 'Open Exploit Injection Workbench' },
  { key: 'A', desc: 'Toggle Enterprise Architecture View' },
  { key: '1 – 6', desc: 'Quick-select scenario preset 1 to 6' },
  { key: '?', desc: 'Toggle keyboard shortcuts cheatsheet' },
  { key: 'Esc', desc: 'Close modals and dismiss inspector drawer' },
];

export function ShortcutsModal() {
  const showShortcuts = useAppStore((s) => s.showShortcuts);
  const setShowShortcuts = useAppStore((s) => s.setShowShortcuts);

  if (!showShortcuts) return null;

  return (
    <div
      className={styles.overlay}
      role="dialog"
      aria-modal="true"
      aria-label="Keyboard Shortcuts"
      onClick={() => setShowShortcuts(false)}
    >
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <span className={styles.title}>⌨️ KEYBOARD SHORTCUTS</span>
          <button
            className={styles.closeBtn}
            onClick={() => setShowShortcuts(false)}
            aria-label="Close modal"
          >
            ✕
          </button>
        </div>
        <div className={styles.list}>
          {SHORTCUTS.map((sc) => (
            <div key={sc.key} className={styles.item}>
              <span>{sc.desc}</span>
              <kbd className={styles.kbd}>{sc.key}</kbd>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
