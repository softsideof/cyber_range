'use client';

// tactical TopBar navigation & simulation controls with live clock and exploit launcher

import React, { useState, useEffect } from 'react';
import { useAppStore } from '@/store';
import { SCENARIOS } from '@/engine/scenarios';
import styles from './TopBar.module.css';

interface TopBarProps {
  onSelectScenario: (id: string) => void;
}

export function TopBar({ onSelectScenario }: TopBarProps) {
  const scenario = useAppStore((s) => s.scenario);
  const isPaused = useAppStore((s) => s.isPaused);
  const togglePaused = useAppStore((s) => s.togglePaused);
  const speed = useAppStore((s) => s.speed);
  const cycleSpeed = useAppStore((s) => s.cycleSpeed);
  const viewMode = useAppStore((s) => s.viewMode);
  const setViewMode = useAppStore((s) => s.setViewMode);
  const autoDemo = useAppStore((s) => s.autoDemo);
  const setAutoDemo = useAppStore((s) => s.setAutoDemo);
  const setShowAttackBuilder = useAppStore((s) => s.setShowAttackBuilder);
  const setShowArchitecture = useAppStore((s) => s.setShowArchitecture);
  const setShowBriefing = useAppStore((s) => s.setShowBriefing);

  const [utcTime, setUtcTime] = useState('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setUtcTime(now.toTimeString().split(' ')[0] + ' UTC');
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <header className={styles.topBar}>
      <div className={styles.leftSection}>
        <span className={styles.brand}>
          <span className={styles.liveDot} />
          CYBERRANGE // SOC COMMAND
        </span>
        <span style={{ color: 'var(--text-3)', fontSize: 10 }}>{utcTime}</span>
        <button
          className={styles.ctrlBtn}
          style={{ fontSize: 10, padding: '3px 8px' }}
          onClick={() => setShowBriefing(true)}
          title="Incident Dossier Briefing"
        >
          BRIEFING
        </button>
      </div>

      <div className={styles.centerSection}>
        <select
          className={styles.scenarioSelect}
          value={scenario?.id || ''}
          onChange={(e) => onSelectScenario(e.target.value)}
        >
          {Object.values(SCENARIOS).map((sc) => (
            <option key={sc.id} value={sc.id}>
              {sc.name} [{sc.difficulty.toUpperCase()}]
            </option>
          ))}
          {scenario && !SCENARIOS[scenario.id] && (
            <option value={scenario.id}>
              {scenario.name} [CUSTOM]
            </option>
          )}
        </select>
      </div>

      <div className={styles.rightSection}>
        <button
          className={`${styles.ctrlBtn} ${styles.hideMobile}`}
          onClick={() => setAutoDemo(!autoDemo)}
          title="Toggle Autonomous Demo Loop"
        >
          <span style={{ color: autoDemo ? 'var(--cyan)' : 'var(--text-3)' }}>●</span>
          AUTO-DEMO: {autoDemo ? 'ENGAGED' : 'MANUAL'}
        </button>

        <button
          className={styles.ctrlBtn}
          onClick={togglePaused}
          title="Space to toggle"
        >
          {isPaused ? '▶ RESUME' : '⏸ PAUSE'}
        </button>

        <button
          className={`${styles.ctrlBtn} ${styles.hideMobile}`}
          onClick={cycleSpeed}
          title="Cycle simulation speed"
        >
          {speed}× CLOCK
        </button>

        <div className={`${styles.divider} ${styles.hideMobile}`} />

        <button
          className={`${styles.ctrlBtn} ${styles.hideMobile}`}
          onClick={() => setViewMode(viewMode === '3d' ? '2d' : '3d')}
          title="Toggle 3D Holographic / 2D Planar Canvas"
        >
          {viewMode === '3d' ? '3D GRID' : '2D MAP'}
        </button>

        <button
          className={`${styles.ctrlBtn} ${styles.hideMobile}`}
          onClick={() => setShowArchitecture(true)}
          title="Architecture Spec (Press A)"
        >
          ARCH [A]
        </button>

        <button
          className={`${styles.ctrlBtn} ${styles.btnWeapon}`}
          onClick={() => setShowAttackBuilder(true)}
          title="Open Weaponized Exploit Workbench (Press B)"
        >
          ⚠️ INJECT EXPLOIT [B]
        </button>
      </div>
    </header>
  );
}
