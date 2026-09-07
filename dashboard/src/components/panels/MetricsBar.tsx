'use client';

// tactical MetricsBar with live sparklines and DEFCON status indicator

import React from 'react';
import { useAppStore } from '@/store';
import { TelemetrySparkline } from './TelemetrySparkline';
import styles from './MetricsBar.module.css';

export function MetricsBar() {
  const metrics = useAppStore((s) => s.metrics);
  const step = useAppStore((s) => s.step);
  const maxSteps = useAppStore((s) => s.maxSteps);

  const getDefconInfo = () => {
    switch (metrics.threatLevel) {
      case 'critical':
      case 'red':
        return { label: 'DEFCON 1 // CRITICAL INTRUSION', cls: styles.defconRed };
      case 'orange':
      case 'yellow':
        return { label: 'DEFCON 3 // ELEVATED THREAT', cls: styles.defconYellow };
      default:
        return { label: 'DEFCON 4 // SYSTEM NORMAL', cls: styles.defconGreen };
    }
  };

  const defcon = getDefconInfo();

  // simulated throughput trend based on step and health
  const ppsValue = Math.max(120, Math.round((100 - metrics.health) * 45 + 180));
  const throughputMb = ((ppsValue * 1.4) / 100).toFixed(1);

  const budgetPct = metrics.maxBudget > 0 ? Math.min(100, (metrics.budget / metrics.maxBudget) * 100) : 0;

  return (
    <div className={styles.container}>
      <div className={`${styles.defconPill} ${defcon.cls}`}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'currentColor' }} />
        {defcon.label}
      </div>

      <div className={styles.divider} />

      <div className={styles.metricItem}>
        <span className={styles.label}>Network Health</span>
        <span className={styles.val} style={{ color: metrics.health >= 80 ? 'var(--green)' : metrics.health >= 50 ? 'var(--yellow)' : 'var(--red)' }}>
          {metrics.health}%
        </span>
        <div className={styles.barTrack}>
          <div
            className={styles.barFill}
            style={{
              width: `${metrics.health}%`,
              backgroundColor: metrics.health >= 80 ? 'var(--green)' : metrics.health >= 50 ? 'var(--yellow)' : 'var(--red)',
            }}
          />
        </div>
      </div>

      <div className={styles.divider} />

      <TelemetrySparkline
        label="Sensor Packets"
        value={ppsValue}
        unit="PPS"
        color="#00f0ff"
        trend={[ppsValue * 0.7, ppsValue * 0.85, ppsValue * 0.78, ppsValue * 0.95, ppsValue]}
      />

      <div className={styles.divider} />

      <TelemetrySparkline
        label="Bandwidth Pipe"
        value={throughputMb}
        unit="MB/s"
        color="#ffb703"
        trend={[12, 18, 15, 24, 32, 28, Number(throughputMb)]}
      />

      <div className={styles.divider} />

      <div className={styles.metricItem}>
        <span className={styles.label}>Action Budget</span>
        <span className={styles.val}>{metrics.budget}/{metrics.maxBudget}</span>
        <div className={styles.barTrack}>
          <div
            className={styles.barFill}
            style={{ width: `${budgetPct}%`, backgroundColor: 'var(--cyan)' }}
          />
        </div>
      </div>

      <div className={styles.divider} />

      <div className={styles.metricItem}>
        <span className={styles.label}>Execution Horizon</span>
        <span className={styles.val}>STEP {step} / {maxSteps}</span>
      </div>

      <div className={styles.divider} />

      <div className={styles.metricItem}>
        <span className={styles.label}>Policy Score</span>
        <span className={styles.val} style={{ color: metrics.score >= 0 ? 'var(--green)' : 'var(--red)' }}>
          {metrics.score >= 0 ? `+${metrics.score.toFixed(2)}` : metrics.score.toFixed(2)}
        </span>
      </div>
    </div>
  );
}
