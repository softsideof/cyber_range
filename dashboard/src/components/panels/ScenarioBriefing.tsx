'use client';

// scenario briefing overlay giving context before simulation defense begins

import React from 'react';
import { useAppStore } from '@/store';
import styles from './ScenarioBriefing.module.css';

interface ScenarioBriefingProps {
  onDismiss: () => void;
}

export function ScenarioBriefing({ onDismiss }: ScenarioBriefingProps) {
  const scenario = useAppStore((s) => s.scenario);

  if (!scenario) return null;

  const diffColorMap: Record<string, string> = {
    easy: 'var(--green)',
    medium: 'var(--yellow)',
    hard: 'var(--orange)',
    nightmare: 'var(--red)',
  };

  const diffColor = diffColorMap[scenario.difficulty] || 'var(--text-1)';

  return (
    <div className={styles.overlay} onClick={onDismiss}>
      <div className={styles.card} onClick={(e) => e.stopPropagation()}>
        <div className={styles.topRow}>
          <span className={styles.title}>{scenario.name}</span>
          <span
            className={styles.diffBadge}
            style={{ color: diffColor, border: `1px solid ${diffColor}` }}
          >
            {scenario.difficulty.toUpperCase()}
          </span>
        </div>

        <div className={styles.desc}>{scenario.description}</div>

        <div className={styles.metaGrid}>
          <div className={styles.metaItem}>
            <span className={styles.metaKey}>Adversary Doctrine</span>
            <span className={styles.metaVal}>{scenario.adversaryBehavior.toUpperCase()}</span>
          </div>
          <div className={styles.metaItem}>
            <span className={styles.metaKey}>Attack Vector Stages</span>
            <span className={styles.metaVal}>{scenario.attackPhases.length} Active Phases</span>
          </div>
          <div className={styles.metaItem}>
            <span className={styles.metaKey}>Telemetry Noise (FPs)</span>
            <span className={styles.metaVal}>{scenario.falsePositiveAlerts?.length || 0} Benign Alerts</span>
          </div>
          <div className={styles.metaItem}>
            <span className={styles.metaKey}>Action Horizon</span>
            <span className={styles.metaVal}>{scenario.maxSteps} Max Steps</span>
          </div>
        </div>

        <div className={styles.metaItem}>
          <span className={styles.metaKey} style={{ marginBottom: 4 }}>MITRE Techniques Tagged</span>
          <div className={styles.mitreRow}>
            {scenario.mitreTechniques.map((t) => (
              <span key={t} className={styles.mitreTag}>{t}</span>
            ))}
          </div>
        </div>

        <button className={styles.dismissBtn} onClick={onDismiss}>
          ENGAGE AUTONOMOUS SOC DEFENSE
        </button>
      </div>
    </div>
  );
}
