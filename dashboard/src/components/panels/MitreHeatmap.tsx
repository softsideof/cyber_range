'use client';

// MITRE ATT&CK matrix coverage panel
// tracks adversary tactics, active techniques, and neutralized vectors

import React from 'react';
import { useAppStore } from '@/store';
import styles from './MitreHeatmap.module.css';

export function MitreHeatmap() {
  const mitre = useAppStore((s) => s.mitre);

  const defendedCount = mitre.filter((m) => m.state === 'defended').length;
  const totalCount = mitre.length;

  // group techniques by tactic
  const grouped = mitre.reduce<Record<string, typeof mitre>>((acc, entry) => {
    const tactic = entry.tactic || 'General';
    if (!acc[tactic]) acc[tactic] = [];
    acc[tactic].push(entry);
    return acc;
  }, {});

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <span>MITRE ATT&CK Matrix</span>
        <span className={styles.counter}>
          Defended: {defendedCount}/{totalCount}
        </span>
      </div>

      <div className={styles.grid}>
        {mitre.length === 0 ? (
          <div className={styles.empty}>No MITRE techniques mapped for this scenario.</div>
        ) : (
          Object.entries(grouped).map(([tactic, items]) => (
            <div key={tactic} className={styles.tacticGroup}>
              <div className={styles.tacticTitle}>{tactic}</div>
              <div className={styles.techniqueList}>
                {items.map((tech) => {
                  let cardClass = styles.cardInactive;
                  let statusClass = styles.statusInactive;
                  if (tech.state === 'attacking') {
                    cardClass = styles.cardAttacking;
                    statusClass = styles.statusAttacking;
                  } else if (tech.state === 'defended') {
                    cardClass = styles.cardDefended;
                    statusClass = styles.statusDefended;
                  }

                  return (
                    <div key={tech.id} className={`${styles.card} ${cardClass}`}>
                      <div className={styles.techId}>{tech.id}</div>
                      <div className={styles.techName} title={tech.name}>{tech.name}</div>
                      <div className={`${styles.techStatus} ${statusClass}`}>
                        {tech.state}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
