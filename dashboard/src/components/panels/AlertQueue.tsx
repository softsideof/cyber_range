'use client';

// enterprise SIEM alert feed with severity filtering and raw packet hex inspector

import React, { useState } from 'react';
import { useAppStore } from '@/store';
import type { Alert } from '@/engine/types';
import styles from './AlertQueue.module.css';

export function AlertQueue() {
  const alerts = useAppStore((s) => s.alerts);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [filterSev, setFilterSev] = useState<'all' | 'critical' | 'high' | 'medium' | 'fp'>('all');

  const filtered = alerts.filter((a) => {
    if (filterSev === 'all') return true;
    if (filterSev === 'fp') return a.isFalsePositive;
    return a.severity === filterSev;
  });

  const getSevClass = (sev: Alert['severity']) => {
    switch (sev) {
      case 'critical': return styles.sevCritical;
      case 'high': return styles.sevHigh;
      case 'medium': return styles.sevMedium;
      case 'low': return styles.sevLow;
      default: return styles.sevInfo;
    }
  };

  // generate simulated packet hex dump for inspection
  const generateHexDump = (alert: Alert) => {
    const hex = [
      '45 00 02 c0 a1 44 40 00 40 06 b8 12',
      'c0 a8 01 64 0a 00 03 01 01 bb d4 31',
      '50 18 0f a0 82 1e 00 00',
    ].join(' ');
    return `[RAW PACKET BUFFER // ${alert.sourceIp}:${alert.type}] ${hex} ... [PAYLOAD SIGNATURE: MATCHED MITRE ${alert.mitreId || 'ANOMALY'}]`;
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.title}>
          <span>SIEM TELEMETRY INGESTION</span>
          <span className={styles.liveTag}>LIVE INTERCEPT</span>
        </div>
        <span style={{ color: 'var(--text-3)', fontSize: 10 }}>
          {alerts.length} PACKETS CAPTURED
        </span>
      </div>

      <div className={styles.filterBar}>
        <button
          className={`${styles.filterBtn} ${filterSev === 'all' ? styles.filterBtnActive : ''}`}
          onClick={() => setFilterSev('all')}
        >
          ALL ({alerts.length})
        </button>
        <button
          className={`${styles.filterBtn} ${filterSev === 'critical' ? styles.filterBtnActive : ''}`}
          onClick={() => setFilterSev('critical')}
        >
          CRITICAL ({alerts.filter((a) => a.severity === 'critical').length})
        </button>
        <button
          className={`${styles.filterBtn} ${filterSev === 'high' ? styles.filterBtnActive : ''}`}
          onClick={() => setFilterSev('high')}
        >
          HIGH ({alerts.filter((a) => a.severity === 'high').length})
        </button>
        <button
          className={`${styles.filterBtn} ${filterSev === 'fp' ? styles.filterBtnActive : ''}`}
          onClick={() => setFilterSev('fp')}
        >
          BENIGN FP ({alerts.filter((a) => a.isFalsePositive).length})
        </button>
      </div>

      <div className={styles.tableWrap}>
        {filtered.length === 0 ? (
          <div className={styles.empty}>
            Awaiting telemetry sensor ingestion. Network perimeter clear.
          </div>
        ) : (
          <table className={styles.table}>
            <thead>
              <tr>
                <th>SEV</th>
                <th>ALERT ID</th>
                <th>ATTACK VECTOR</th>
                <th>TARGET</th>
                <th>STATUS</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((alert) => {
                const isExpanded = expandedId === alert.alertId;
                return (
                  <React.Fragment key={alert.alertId}>
                    <tr
                      className={`${styles.row} ${isExpanded ? styles.expanded : ''}`}
                      onClick={() => setExpandedId(isExpanded ? null : alert.alertId)}
                    >
                      <td className={`${styles.cell} ${getSevClass(alert.severity)}`}>
                        {alert.severity.toUpperCase()}
                      </td>
                      <td className={`${styles.cell} font-mono`} style={{ color: 'var(--cyan)' }}>
                        {alert.alertId}
                      </td>
                      <td className={styles.cell}>
                        <span style={{ color: '#fff', fontWeight: 600 }}>{alert.title}</span>
                      </td>
                      <td className={`${styles.cell} font-mono`} style={{ color: 'var(--accent)' }}>
                        {alert.targetNodeId}
                      </td>
                      <td className={styles.cell}>
                        <span style={{ fontSize: 10, color: alert.status === 'contained' ? 'var(--green)' : alert.status === 'dismissed' ? 'var(--text-3)' : 'var(--red)' }}>
                          ● {alert.status.toUpperCase()}
                        </span>
                      </td>
                    </tr>

                    {isExpanded && (
                      <tr className={styles.detailRow}>
                        <td colSpan={5}>
                          <div className={styles.detailContent}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10.5 }}>
                              <span><strong>SOURCE IP:</strong> <span style={{ color: 'var(--red)' }}>{alert.sourceIp}</span></span>
                              <span><strong>MITRE TECHNIQUE:</strong> <span style={{ color: 'var(--accent)' }}>{alert.mitreId || 'N/A'}</span></span>
                              <span><strong>CONFIDENCE:</strong> {(alert.confidence * 100).toFixed(0)}%</span>
                            </div>

                            <div className={styles.hexSnippet}>
                              {generateHexDump(alert)}
                            </div>

                            <div className={styles.evidenceBox}>
                              <strong>FORENSIC SIGNATURE:</strong> {alert.forensicEvidence}
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
