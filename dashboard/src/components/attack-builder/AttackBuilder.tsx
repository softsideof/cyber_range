'use client';

// Weaponized Adversarial Exploit Workbench
// features threat actor intelligence profiles, kill-chain routing, and bespoke zero-day payload constructor

import React, { useState } from 'react';
import { useAppStore } from '@/store';
import { VIRUS_TEMPLATES } from '@/engine/attack-builder';
import type { CustomAttackConfig, AlertType, Difficulty } from '@/engine/types';
import { NodeSelector } from './NodeSelector';
import styles from './AttackBuilder.module.css';

interface AttackBuilderProps {
  onDeployCustom: (config: CustomAttackConfig) => void;
}

const THREAT_ACTORS: Record<string, { actor: string; cve: string; signature: string }> = {
  wannacry_lite: {
    actor: 'LAZARUS GROUP // WORM OUTBREAK',
    cve: 'CVE-2017-0144 (ETERNALBLUE)',
    signature: 'W32.WannaCrypt.Worm.Payload',
  },
  solarwinds_jr: {
    actor: 'APT-29 (COZY BEAR) // SUPPLY CHAIN',
    cve: 'CVE-2020-10148 (SUNBURST)',
    signature: 'SolarWinds.Orion.Core.BusinessLayer.dll',
  },
  ghost_operator: {
    actor: 'EQUATION GROUP // DUAL APT INSIDER',
    cve: 'ZERO-DAY TOKEN THEFT & SENSITIVE EXFIL',
    signature: 'CloudToken.Hijack.DirectExfil.APT',
  },
};

export function AttackBuilder({ onDeployCustom }: AttackBuilderProps) {
  const showAttackBuilder = useAppStore((s) => s.showAttackBuilder);
  const setShowAttackBuilder = useAppStore((s) => s.setShowAttackBuilder);

  const [activeTab, setActiveTab] = useState<'templates' | 'custom'>('templates');

  // custom attack form state
  const [name, setName] = useState('Operation Zero-Day Red');
  const [attackVector, setAttackVector] = useState<AlertType>('ransomware');
  const [difficulty, setDifficulty] = useState<Difficulty>('hard');
  const [maxSteps, setMaxSteps] = useState(25);
  const [selectedTargets, setSelectedTargets] = useState<string[]>(['ws-01', 'ws-02', 'backup-01']);
  const [rotateC2, setRotateC2] = useState(true);
  const [recompromise, setRecompromise] = useState(true);
  const [decoyAlerts, setDecoyAlerts] = useState(true);

  if (!showAttackBuilder) return null;

  const toggleTarget = (nodeId: string) => {
    setSelectedTargets((prev) =>
      prev.includes(nodeId) ? prev.filter((id) => id !== nodeId) : [...prev, nodeId],
    );
  };

  const handleDeployTemplate = (templateKey: string) => {
    const template = VIRUS_TEMPLATES[templateKey];
    if (template) {
      onDeployCustom(template);
      setShowAttackBuilder(false);
    }
  };

  const handleDeployCustom = () => {
    if (selectedTargets.length === 0) {
      alert('Select at least one host target on the network map.');
      return;
    }

    const phases = selectedTargets.map((target, idx) => ({
      name: `Stage ${idx + 1}: ${attackVector.toUpperCase()} Lateral Injection`,
      target,
      type: attackVector,
      mitreId: attackVector === 'ransomware' ? 'T1486' : attackVector === 'exfiltration' ? 'T1041' : 'T1021',
    }));

    const config: CustomAttackConfig = {
      name,
      description: `Bespoke adversarial cyber weapon exercising ${attackVector} targeting ${selectedTargets.join(', ')}.`,
      attackVector,
      targets: selectedTargets,
      evasion: {
        rotateC2,
        recompromise,
        decoyAlerts,
      },
      difficulty,
      maxSteps,
      phases,
    };

    onDeployCustom(config);
    setShowAttackBuilder(false);
  };

  return (
    <div className={styles.overlay} onClick={() => setShowAttackBuilder(false)}>
      <div className={styles.drawer} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <span className={styles.title}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--red)', boxShadow: '0 0 8px var(--red)' }} />
            WEAPONIZED ADVERSARY EXPLOIT WORKBENCH
          </span>
          <button className={styles.closeBtn} onClick={() => setShowAttackBuilder(false)}>
            ESC
          </button>
        </div>

        <div className={styles.tabNav}>
          <button
            className={`${styles.tabBtn} ${activeTab === 'templates' ? styles.tabBtnActive : ''}`}
            onClick={() => setActiveTab('templates')}
          >
            Weaponized APT Arsenals
          </button>
          <button
            className={`${styles.tabBtn} ${activeTab === 'custom' ? styles.tabBtnActive : ''}`}
            onClick={() => setActiveTab('custom')}
          >
            Zero-Day Exploit Composer
          </button>
        </div>

        <div className={styles.body}>
          {activeTab === 'templates' ? (
            <>
              <div style={{ fontSize: 11, color: 'var(--text-3)', letterSpacing: '0.04em' }}>
                SELECT AN ADVERSARIAL THREAT CAMPAIGN TO INITIATE LIVE ENGAGEMENT:
              </div>

              {Object.entries(VIRUS_TEMPLATES).map(([key, template]) => {
                const intel = THREAT_ACTORS[key];
                return (
                  <div
                    key={key}
                    className={styles.templateCard}
                    onClick={() => handleDeployTemplate(key)}
                  >
                    <div className={styles.cardTop}>
                      <span className={styles.actorTag}>{intel?.actor || 'NATION STATE APT'}</span>
                      <span style={{ color: 'var(--red)', fontSize: 10, fontWeight: 700 }}>
                        {template.difficulty.toUpperCase()}
                      </span>
                    </div>

                    <div className={styles.templateName}>{template.name}</div>
                    <div className={styles.templateDesc}>{template.description}</div>

                    <div className={styles.killChainRow}>
                      <span className={`${styles.chainStep} ${styles.chainActive}`}>[1. ACCESS]</span>
                      <span style={{ color: 'var(--text-3)', fontSize: 8 }}>➔</span>
                      <span className={`${styles.chainStep} ${styles.chainActive}`}>[2. LATERAL WORM]</span>
                      <span style={{ color: 'var(--text-3)', fontSize: 8 }}>➔</span>
                      <span className={`${styles.chainStep} ${styles.chainActive}`}>[3. ESCALATION]</span>
                      <span style={{ color: 'var(--text-3)', fontSize: 8 }}>➔</span>
                      <span className={`${styles.chainStep} ${styles.chainActive}`}>[4. TARGET IMPACT]</span>
                    </div>

                    <div style={{ fontSize: 9.5, color: 'var(--text-3)', marginTop: 2 }}>
                      SIGNATURE: <span style={{ color: 'var(--cyan)' }}>{intel?.signature}</span>
                    </div>
                  </div>
                );
              })}
            </>
          ) : (
            <>
              <div className={styles.formGroup}>
                <label className={styles.label}>Campaign Designation</label>
                <input
                  className={styles.input}
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>

              <div className={styles.formGroup}>
                <label className={styles.label}>Weaponized Payload Delivery Vector</label>
                <select
                  className={styles.select}
                  value={attackVector}
                  onChange={(e) => setAttackVector(e.target.value as AlertType)}
                >
                  <option value="ransomware">Ransomware Crypter (AES-256 Worm)</option>
                  <option value="exfiltration">C2 Covert Exfiltration Channel</option>
                  <option value="malware">Trojan Supply Chain Memory Implant</option>
                  <option value="lateral_movement">SMB EternalBlue Protocol Propagation</option>
                  <option value="privilege_escalation">Kerberoasting & AD Token Theft</option>
                </select>
              </div>

              <div className={styles.formGroup}>
                <label className={styles.label}>Host Targets ({selectedTargets.length} Targeted)</label>
                <NodeSelector
                  selectedNodes={selectedTargets}
                  onToggleNode={toggleTarget}
                />
              </div>

              <div className={styles.formGroup}>
                <label className={styles.label}>Adversarial Anti-Analysis & Evasion</label>
                <div className={styles.checkboxGroup}>
                  <label className={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      checked={rotateC2}
                      onChange={(e) => setRotateC2(e.target.checked)}
                    />
                    <span>Evasive C2 IP Fast-Flux Mutation</span>
                  </label>
                  <label className={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      checked={recompromise}
                      onChange={(e) => setRecompromise(e.target.checked)}
                    />
                    <span>Persistent Dormant Backdoors (Re-infects Patches)</span>
                  </label>
                  <label className={styles.checkboxLabel}>
                    <input
                      type="checkbox"
                      checked={decoyAlerts}
                      onChange={(e) => setDecoyAlerts(e.target.checked)}
                    />
                    <span>Deploy Decoy Telemetry & False-Positive Chaff</span>
                  </label>
                </div>
              </div>
            </>
          )}
        </div>

        {activeTab === 'custom' && (
          <div className={styles.footer}>
            <button className={styles.deployBtn} onClick={handleDeployCustom}>
              ARM & INJECT ZERO-DAY ADVERSARIAL WEAPON
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
