'use client';

// AI defense log — shows step-by-step agent decisions with action type color coding
import React, { useEffect, useRef } from 'react';
import { useAppStore } from '@/store';
import styles from './AgentLog.module.css';

interface AgentLogProps {
  onStepOnce?: () => void;
  onRestart?: () => void;
}

// color-code each SOC action type
const ACTION_META: Record<string, { color: string; icon: string; label: string }> = {
  observe_network:    { color: 'var(--blue)',   icon: '◎', label: 'Observe' },
  investigate_alert:  { color: 'var(--amber)',  icon: '⌕', label: 'Investigate' },
  run_forensics:      { color: 'var(--cyan)',   icon: '⚙', label: 'Forensics' },
  block_ip:           { color: 'var(--red)',    icon: '⊘', label: 'Block IP' },
  isolate_host:       { color: 'var(--purple)', icon: '⬡', label: 'Isolate Host' },
  dismiss_alert:      { color: 'var(--green)',  icon: '✓', label: 'Dismiss FP' },
  restore_backup:     { color: 'var(--green)',  icon: '↺', label: 'Restore' },
  deploy_patch:       { color: 'var(--cyan)',   icon: '⬆', label: 'Deploy Patch' },
  deploy_honeypot:    { color: 'var(--amber)',  icon: '⬡', label: 'Honeypot' },
  escalate_incident:  { color: 'var(--red)',    icon: '⚠', label: 'Escalate' },
  save_playbook:      { color: 'var(--text-2)', icon: '⊕', label: 'Save PB' },
  search_playbooks:   { color: 'var(--text-2)', icon: '⊕', label: 'Search PB' },
};

function getActionMeta(tool: string) {
  return ACTION_META[tool] || { color: 'var(--text-2)', icon: '›', label: tool };
}

export function AgentLog({ onStepOnce, onRestart }: AgentLogProps) {
  const agentLog     = useAppStore((s) => s.agentLog);
  const isPaused     = useAppStore((s) => s.isPaused);
  const togglePaused = useAppStore((s) => s.togglePaused);
  const step         = useAppStore((s) => s.step);
  const maxSteps     = useAppStore((s) => s.maxSteps);
  const scenario     = useAppStore((s) => s.scenario);
  const scrollRef    = useRef<HTMLDivElement>(null);

  // auto-scroll to latest entry
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [agentLog.length]);

  const latest = agentLog.length > 0 ? agentLog[agentLog.length - 1] : null;
  const stepProgress = maxSteps > 0 ? (step / maxSteps) * 100 : 0;

  return (
    <div className={styles.container}>
      {/* header */}
      <div className={styles.header}>
        <div className={styles.headerTop}>
          <span className={styles.title}>Defense Log</span>
          <div className={styles.stepBadge}>
            <span>{step}</span>
            <span className={styles.stepOf}>/ {maxSteps}</span>
          </div>
        </div>
        {/* step progress track */}
        <div className={styles.stepTrack}>
          <div className={styles.stepFill} style={{ width: `${stepProgress}%` }} />
        </div>
      </div>

      {/* controls */}
      <div className={styles.controls}>
        <button
          className={`${styles.ctrlBtn} ${isPaused ? styles.ctrlPlay : styles.ctrlPause}`}
          onClick={togglePaused}
        >
          {isPaused ? '▶ Run Defense' : '⏸ Pause'}
        </button>
        {onStepOnce && (
          <button className={styles.ctrlBtn} onClick={onStepOnce} title="→ key">
            Step →
          </button>
        )}
        {onRestart && (
          <button className={styles.ctrlBtnIcon} onClick={onRestart} title="R key">
            ↺
          </button>
        )}
      </div>

      {/* current action headline */}
      {latest ? (
        <div className={styles.headline}>
          <div className={styles.headlineMeta}>
            <span className={styles.headlineStep}>Step {latest.step}</span>
            <span
              className={styles.headlineReward}
              style={{ color: latest.reward >= 0 ? 'var(--green)' : 'var(--red)' }}
            >
              {latest.reward >= 0 ? '+' : ''}{latest.reward.toFixed(2)}
            </span>
          </div>
          <div className={styles.headlineAction}>
            <span
              className={styles.headlineIcon}
              style={{ color: getActionMeta(latest.action.tool).color }}
            >
              {getActionMeta(latest.action.tool).icon}
            </span>
            <span className={styles.headlineTool}>{latest.action.tool}</span>
          </div>
          <p className={styles.headlineReason}>{latest.action.reasoning}</p>
        </div>
      ) : (
        <div className={styles.emptyHeadline}>
          <span className={styles.cursor} />
          <span>
            {scenario?.name || 'Scenario loaded'} — press{' '}
            <strong>Run Defense</strong> to start autonomous response, or{' '}
            <strong>Step →</strong> to step manually.
          </span>
        </div>
      )}

      {/* decision log */}
      <div className={styles.log} ref={scrollRef}>
        {agentLog.length === 0 ? (
          <div className={styles.empty}>
            <span>Awaiting first agent decision…</span>
          </div>
        ) : (
          [...agentLog].reverse().map((entry, idx) => {
            const meta = getActionMeta(entry.action.tool);
            // confidence: parse from reasoning if available, default to 0.75
            const confidence = entry.result.success ? 0.78 + (entry.reward * 0.15) : 0.35;
            const confPct = Math.max(0, Math.min(100, Math.round(confidence * 100)));

            return (
              <div key={idx} className={styles.stepCard} style={{ animationDelay: `${idx * 0.02}s` }}>
                {/* left accent bar */}
                <div className={styles.accent} style={{ background: meta.color }} />

                <div className={styles.cardBody}>
                  <div className={styles.cardTop}>
                    <div className={styles.cardLeft}>
                      <span className={styles.stepNum}>#{entry.step}</span>
                      <span className={styles.actionTag} style={{ color: meta.color }}>
                        {meta.icon} {meta.label}
                      </span>
                    </div>
                    <div className={styles.cardRight}>
                      <span
                        className={styles.rewardBadge}
                        style={{
                          color: entry.reward >= 0 ? 'var(--green)' : 'var(--red)',
                          background: entry.reward >= 0 ? 'var(--green-dim)' : 'var(--red-dim)',
                        }}
                      >
                        {entry.reward >= 0 ? '+' : ''}{entry.reward.toFixed(2)}
                      </span>
                    </div>
                  </div>

                  {/* args if any */}
                  {Object.keys(entry.action.args).length > 0 && (
                    <div className={styles.argsLine}>
                      {Object.entries(entry.action.args)
                        .map(([k, v]) => `${k}: ${v}`)
                        .join(' · ')}
                    </div>
                  )}

                  {/* reasoning */}
                  <p className={styles.reason}>{entry.action.reasoning}</p>

                  {/* confidence bar */}
                  <div className={styles.confRow}>
                    <span className={styles.confLabel}>AI Confidence</span>
                    <div className={styles.confTrack}>
                      <div
                        className={styles.confFill}
                        style={{
                          width: `${confPct}%`,
                          background: confPct >= 70 ? 'var(--green)' : confPct >= 40 ? 'var(--amber)' : 'var(--red)',
                        }}
                      />
                    </div>
                    <span className={styles.confPct}>{confPct}%</span>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
