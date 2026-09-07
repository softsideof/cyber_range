'use client';

// episode results scorecard with 5-component weighted metrics and CyberJudge persona reviews

import React from 'react';
import { useAppStore } from '@/store';
import { generateIncidentReport, downloadIncidentReport } from '@/engine/reporter';
import styles from './ScoreCard.module.css';

interface ScoreCardProps {
  onRestart?: () => void;
  onOpenBuilder?: () => void;
  onNext?: () => void;
}

export function ScoreCard({ onRestart, onOpenBuilder, onNext }: ScoreCardProps) {
  const graderResult = useAppStore((s) => s.graderResult);
  const scenario = useAppStore((s) => s.scenario);
  const agentLog = useAppStore((s) => s.agentLog);
  const topology = useAppStore((s) => s.topology);

  const handleDownloadReport = () => {
    if (!graderResult) return;
    const reportMd = generateIncidentReport(graderResult, scenario, agentLog, topology);
    const filename = `Incident_Report_${scenario?.id || 'sim'}.md`;
    downloadIncidentReport(reportMd, filename);
  };

  if (!graderResult) {
    return (
      <div className={styles.container}>
        <div className={styles.header}>
          <span className={styles.title}>Episode In Progress</span>
        </div>
        <p className="text-muted" style={{ padding: '20px 0' }}>
          Final grade and CyberJudge verdicts will appear once the attack scenario concludes or all threats are neutralized.
        </p>
      </div>
    );
  }

  const scorePct = Math.round(graderResult.finalScore * 100);
  let gradeLetter = 'F';
  if (scorePct >= 90) gradeLetter = 'A+';
  else if (scorePct >= 80) gradeLetter = 'A';
  else if (scorePct >= 70) gradeLetter = 'B';
  else if (scorePct >= 60) gradeLetter = 'C';
  else if (scorePct >= 50) gradeLetter = 'D';

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <span className={styles.title}>Incident Response Evaluation</span>
        <span className="text-dim font-mono">{scenario?.name}</span>
      </div>

      <div className={styles.scoreBanner}>
        <div className={styles.gradeBox}>
          <span className={styles.gradeLetter}>{gradeLetter}</span>
          <span className={styles.gradeLabel}>IR Readiness Grade</span>
        </div>
        <div className={styles.scoreValueBox}>
          <span className={styles.finalScoreNum}>{(graderResult.finalScore * 100).toFixed(1)}%</span>
          <span className={styles.gradeLabel}>Weighted Performance Score</span>
        </div>
      </div>

      <table className={styles.breakdownTable}>
        <thead>
          <tr>
            <th>Component</th>
            <th>Weight</th>
            <th>Score</th>
            <th>Outcome</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Threat Response</td>
            <td>35%</td>
            <td>{(graderResult.threatResponse * 100).toFixed(1)}%</td>
            <td>{graderResult.details.threatsNeutralized} / {graderResult.details.totalThreats} Neutralized</td>
          </tr>
          <tr>
            <td>False Positive Triage</td>
            <td>20%</td>
            <td>{(graderResult.falsePositiveHandling * 100).toFixed(1)}%</td>
            <td>{graderResult.details.fpDismissed} / {graderResult.details.totalFps} FPs Handled</td>
          </tr>
          <tr>
            <td>Data Protection</td>
            <td>20%</td>
            <td>{(graderResult.dataProtection * 100).toFixed(1)}%</td>
            <td>{graderResult.details.dataExfiltratedMb} MB Exfiltrated</td>
          </tr>
          <tr>
            <td>Collateral Damage</td>
            <td>15%</td>
            <td>{(graderResult.collateralDamage * 100).toFixed(1)}%</td>
            <td>{graderResult.details.healthyHostsIsolated} Clean Hosts Cut</td>
          </tr>
          <tr>
            <td>Operational Efficiency</td>
            <td>10%</td>
            <td>{(graderResult.efficiency * 100).toFixed(1)}%</td>
            <td>{graderResult.details.stepsUsed} / {graderResult.details.maxSteps} Steps Used</td>
          </tr>
        </tbody>
      </table>

      <div className={styles.judgeSection}>
        <div className={styles.judgeHeading}>CyberJudge Persona Verdicts</div>
        <div className={styles.judgeCard}>
          <div className={styles.judgeName}>Junior SOC Analyst ({Math.round(graderResult.judgeVerdicts.junior.score * 100)}%)</div>
          <div className={styles.judgeVerdict}>"{graderResult.judgeVerdicts.junior.verdict}"</div>
        </div>
        <div className={styles.judgeCard}>
          <div className={styles.judgeName}>Senior SOC Lead ({Math.round(graderResult.judgeVerdicts.senior.score * 100)}%)</div>
          <div className={styles.judgeVerdict}>"{graderResult.judgeVerdicts.senior.verdict}"</div>
        </div>
        <div className={styles.judgeCard}>
          <div className={styles.judgeName}>Incident Commander ({Math.round(graderResult.judgeVerdicts.commander.score * 100)}%)</div>
          <div className={styles.judgeVerdict}>"{graderResult.judgeVerdicts.commander.verdict}"</div>
        </div>
      </div>

      <div className={styles.actionRow}>
        <button
          className={styles.btnPrimary}
          onClick={handleDownloadReport}
          title="Export Markdown Incident Report"
        >
          📄 Export AAR Report
        </button>
        {onRestart && (
          <button className={styles.btnSecondary} onClick={onRestart}>
            Replay Scenario
          </button>
        )}
        {onOpenBuilder && (
          <button className={styles.btnSecondary} onClick={onOpenBuilder}>
            Inject Custom Virus
          </button>
        )}
        {onNext && (
          <button className={styles.btnSecondary} onClick={onNext}>
            Next Scenario
          </button>
        )}
      </div>
    </div>
  );
}
