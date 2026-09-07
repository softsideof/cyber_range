'use client';

// live metrics bar — DEFCON, health, step progress, score
import React from 'react';
import { useAppStore } from '@/store';
import styles from './MetricsBar.module.css';

export function MetricsBar() {
  const metrics  = useAppStore((s) => s.metrics);
  const step     = useAppStore((s) => s.step);
  const maxSteps = useAppStore((s) => s.maxSteps);

  // derive DEFCON level 1-5 from threat
  const defconLevel = () => {
    switch (metrics.threatLevel) {
      case 'critical': return 1;
      case 'red':      return 2;
      case 'orange':   return 3;
      case 'yellow':   return 4;
      default:         return 5;
    }
  };

  const defcon = defconLevel();
  const stepPct = maxSteps > 0 ? Math.min(100, (step / maxSteps) * 100) : 0;
  const healthColor = metrics.health >= 75 ? 'var(--green)' : metrics.health >= 40 ? 'var(--amber)' : 'var(--red)';
  const scoreSigned = metrics.score >= 0
    ? `+${metrics.score.toFixed(2)}`
    : metrics.score.toFixed(2);

  // simulated network traffic based on health
  const pps = Math.round((100 - metrics.health) * 38 + 200);

  return (
    <div className={styles.bar}>
      {/* DEFCON strip — 5 numbered segments */}
      <div className={styles.section}>
        <span className={styles.label}>DEFCON</span>
        <div className={styles.defconStrip}>
          {[1, 2, 3, 4, 5].map((n) => (
            <div
              key={n}
              className={styles.defconSeg}
              data-active={defcon <= n ? 'true' : undefined}
              data-level={n}
              title={`DEFCON ${n}`}
            >
              {n}
            </div>
          ))}
        </div>
      </div>

      <div className={styles.sep} />

      {/* Network health */}
      <div className={styles.section}>
        <span className={styles.label}>Network Health</span>
        <div className={styles.healthRow}>
          <span className={styles.healthVal} style={{ color: healthColor }}>
            {metrics.health}%
          </span>
          <div className={styles.healthTrack}>
            <div
              className={styles.healthFill}
              style={{ width: `${metrics.health}%`, background: healthColor }}
            />
          </div>
        </div>
      </div>

      <div className={styles.sep} />

      {/* Step progress */}
      <div className={styles.section}>
        <span className={styles.label}>Step Progress</span>
        <div className={styles.healthRow}>
          <span className={styles.healthVal}>{step}<span className={styles.stepMax}> / {maxSteps}</span></span>
          <div className={styles.healthTrack}>
            <div
              className={styles.healthFill}
              style={{ width: `${stepPct}%`, background: 'var(--blue)' }}
            />
          </div>
        </div>
      </div>

      <div className={styles.sep} />

      {/* Traffic sensor */}
      <div className={styles.section}>
        <span className={styles.label}>Network Traffic</span>
        <span className={styles.monoVal} style={{ color: 'var(--cyan)' }}>
          {pps} <span className={styles.unit}>PPS</span>
        </span>
      </div>

      <div className={styles.sep} />

      {/* Policy score */}
      <div className={styles.section}>
        <span className={styles.label}>Policy Score</span>
        <span
          className={styles.monoVal}
          style={{ color: metrics.score >= 0 ? 'var(--green)' : 'var(--red)' }}
        >
          {scoreSigned}
        </span>
      </div>

      <div className={styles.sep} />

      {/* Action budget */}
      <div className={styles.section}>
        <span className={styles.label}>Action Budget</span>
        <span className={styles.monoVal}>
          {metrics.budget}
          <span className={styles.unit}> / {metrics.maxBudget}</span>
        </span>
      </div>
    </div>
  );
}
