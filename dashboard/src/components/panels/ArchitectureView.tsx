'use client';

// architecture view modal explaining OpenEnv RL training loop and client-side simulation port

import React from 'react';
import { useAppStore } from '@/store';
import styles from './ArchitectureView.module.css';

export function ArchitectureView() {
  const showArchitecture = useAppStore((s) => s.showArchitecture);
  const setShowArchitecture = useAppStore((s) => s.setShowArchitecture);

  if (!showArchitecture) return null;

  return (
    <div className={styles.overlay} onClick={() => setShowArchitecture(false)}>
      <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
        <div className={styles.header}>
          <span className={styles.title}>System Architecture & AI Engineering Spec</span>
          <button className={styles.closeBtn} onClick={() => setShowArchitecture(false)}>
            ESC
          </button>
        </div>

        <div className={styles.content}>
          <div className={styles.diagBox}>
            <div style={{ color: 'var(--accent)', fontWeight: 700, fontSize: 11 }}>
              OPENENV REINFORCEMENT LEARNING & SIMULATION ENGINE PIPELINE
            </div>
            <div className={styles.diagGrid}>
              <div className={styles.diagCard}>
                <span className={styles.cardTag}>1. Environment</span>
                <span className={styles.cardTitle}>12-Node Enterprise Topology</span>
                <span className={styles.cardBody}>
                  Simulates multi-tier enterprise network with DMZ, Active Directory domain controller, database, app servers, endpoints, and honeypot traps.
                </span>
              </div>
              <div className={styles.diagCard}>
                <span className={styles.cardTag}>2. Adversary</span>
                <span className={styles.cardTitle}>Dynamic Attack Progression</span>
                <span className={styles.cardBody}>
                  Simulates APTs with 4 distinct behavioral modes: static, evasive (rotates C2 IPs), persistent (re-infects backdoors), and adaptive (decoy injections).
                </span>
              </div>
              <div className={styles.diagCard}>
                <span className={styles.cardTag}>3. Defender</span>
                <span className={styles.cardTitle}>Autonomous SOC Policy</span>
                <span className={styles.cardBody}>
                  Trained via Group Relative Policy Optimization (GRPO) to balance triage speed, investigation depth, collateral containment, and FP handling.
                </span>
              </div>
              <div className={styles.diagCard}>
                <span className={styles.cardTag}>4. Telemetry</span>
                <span className={styles.cardTitle}>SIEM & Forensic Evidence</span>
                <span className={styles.cardBody}>
                  Generates RFC 5737 compliant telemetry, genuine CVE identifiers, and realistic forensic artifacts differentiating malicious activity from noise.
                </span>
              </div>
              <div className={styles.diagCard}>
                <span className={styles.cardTag}>5. Reward Engine</span>
                <span className={styles.cardTitle}>5-Objective Reward Shaping</span>
                <span className={styles.cardBody}>
                  Weighted multi-objective reward function preventing reward hacking: penalizes action-spamming, healthy host isolation, and uninvestigated dismissal.
                </span>
              </div>
              <div className={styles.diagCard}>
                <span className={styles.cardTag}>6. CyberJudge</span>
                <span className={styles.cardTitle}>Multi-Persona Evaluation</span>
                <span className={styles.cardBody}>
                  Synthesizes operational readiness grades with 3 perspectives: Junior Analyst (discipline), Senior Lead (triage efficiency), Commander (business impact).
                </span>
              </div>
            </div>
          </div>

          <div className={styles.notesSection}>
            <div className={styles.notesHeading}>Key Resume & Interview Talking Points</div>
            <div className={styles.bullet}>
              <strong>Dual-Track AI Engineering:</strong> Developed the RL training environment in Python conforming to the OpenEnv standard, then created a 100% client-side TypeScript simulation engine capable of running in Web Workers at 60 FPS without server requirements.
            </div>
            <div className={styles.bullet}>
              <strong>No Hallucinations, Deterministic Parity:</strong> All mathematical reward calculations, adversary state machines, and network topologies have identical behavior between the Python and TypeScript runtimes.
            </div>
            <div className={styles.bullet}>
              <strong>Interactive Virus Customization:</strong> Interviewers can compose bespoke multi-stage attacks or test pre-built weaponized templates (WannaCry SMB worm, SolarWinds trojan, Ghost Operator insider APT) and watch the AI defend in real-time.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
