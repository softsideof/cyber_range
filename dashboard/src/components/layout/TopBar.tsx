'use client';

// clean tactical top-bar — scenario selector, status pill, playback controls
import React, { useState, useEffect } from 'react';
import { useAppStore } from '@/store';
import { SCENARIOS } from '@/engine/scenarios';
import styles from './TopBar.module.css';

interface TopBarProps {
  onSelectScenario: (id: string) => void;
}

export function TopBar({ onSelectScenario }: TopBarProps) {
  const scenario     = useAppStore((s) => s.scenario);
  const isPaused     = useAppStore((s) => s.isPaused);
  const togglePaused = useAppStore((s) => s.togglePaused);
  const speed        = useAppStore((s) => s.speed);
  const cycleSpeed   = useAppStore((s) => s.cycleSpeed);
  const metrics      = useAppStore((s) => s.metrics);
  const mode         = useAppStore((s) => s.mode);
  const setShowAttackBuilder = useAppStore((s) => s.setShowAttackBuilder);
  const setShowBriefing      = useAppStore((s) => s.setShowBriefing);
  const setShowShortcuts     = useAppStore((s) => s.setShowShortcuts);

  const [utcTime, setUtcTime] = useState('');

  useEffect(() => {
    const tick = () => {
      const d = new Date();
      const h = String(d.getUTCHours()).padStart(2, '0');
      const m = String(d.getUTCMinutes()).padStart(2, '0');
      const s = String(d.getUTCSeconds()).padStart(2, '0');
      setUtcTime(`${h}:${m}:${s} UTC`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  // status pill based on threat level
  const getStatusPill = () => {
    if (mode === 'complete') return { label: 'SIMULATION COMPLETE', cls: styles.pillComplete };
    if (metrics.threatLevel === 'critical' || metrics.threatLevel === 'red')
      return { label: '⚡ ACTIVE BREACH', cls: styles.pillBreach };
    if (metrics.threatLevel === 'orange' || metrics.threatLevel === 'yellow')
      return { label: '⚠ ELEVATED THREAT', cls: styles.pillElevated };
    return { label: '● SYSTEM NOMINAL', cls: styles.pillNormal };
  };

  const pill = getStatusPill();

  return (
    <header className={styles.topBar}>
      {/* left — brand + time + status */}
      <div className={styles.left}>
        <span className={styles.brand}>CyberRange</span>
        <span className={styles.clock}>{utcTime}</span>
        <span className={`${styles.statusPill} ${pill.cls}`}>{pill.label}</span>
      </div>

      {/* center — scenario selector */}
      <div className={styles.center}>
        <label className={styles.scenarioLabel}>Scenario</label>
        <select
          className={styles.scenarioSelect}
          value={scenario?.id || ''}
          onChange={(e) => onSelectScenario(e.target.value)}
        >
          {Object.values(SCENARIOS).map((sc) => (
            <option key={sc.id} value={sc.id}>
              {sc.name} · {sc.difficulty.toUpperCase()}
            </option>
          ))}
          {scenario && !SCENARIOS[scenario.id] && (
            <option value={scenario.id}>{scenario.name} · CUSTOM</option>
          )}
        </select>
      </div>

      {/* right — playback + actions */}
      <div className={styles.right}>
        <button
          className={`${styles.btn} ${isPaused ? styles.btnPlay : styles.btnPause}`}
          onClick={togglePaused}
          title="Space to toggle"
        >
          {isPaused ? '▶ Run' : '⏸ Pause'}
        </button>

        <button className={styles.btn} onClick={cycleSpeed} title="Cycle speed">
          {speed}×
        </button>

        <div className={styles.divider} />

        <button className={styles.btn} onClick={() => setShowBriefing(true)} title="Scenario briefing">
          Briefing
        </button>

        <button
          className={`${styles.btn} ${styles.btnDanger}`}
          onClick={() => setShowAttackBuilder(true)}
          title="Inject adversary campaign (B)"
        >
          Inject Threat
        </button>

        <button
          className={`${styles.btn} ${styles.btnIcon}`}
          onClick={() => setShowShortcuts(true)}
          title="Keyboard shortcuts (?)"
          aria-label="Keyboard shortcuts"
        >
          ?
        </button>
      </div>
    </header>
  );
}
