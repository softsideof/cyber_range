'use client';

// interactive incident story and AI defense feed (Linear / Vercel style)
// provides step-by-step playback controls, plain English explanations, and decision rationale

import React, { useEffect, useRef } from 'react';
import { useAppStore } from '@/store';
import styles from './AgentLog.module.css';

interface AgentLogProps {
  onStepOnce?: () => void;
  onRestart?: () => void;
}

export function AgentLog({ onStepOnce, onRestart }: AgentLogProps) {
  const agentLog = useAppStore((s) => s.agentLog);
  const isPaused = useAppStore((s) => s.isPaused);
  const togglePaused = useAppStore((s) => s.togglePaused);
  const step = useAppStore((s) => s.step);
  const maxSteps = useAppStore((s) => s.maxSteps);
  const scenario = useAppStore((s) => s.scenario);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [agentLog.length]);

  const latestEntry = agentLog.length > 0 ? agentLog[agentLog.length - 1] : null;

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.titleRow}>
          <span className={styles.title}>
            🤖 AUTONOMOUS AI SOC DEFENSE LOG
          </span>
          <span style={{ fontSize: 11, color: '#8b949e', fontFamily: 'var(--font-mono)' }}>
            STEP {step} OF {maxSteps}
          </span>
        </div>

        {/* INTERVIEWER STEPPING & PLAYBACK CONTROLS */}
        <div className={styles.controlBar}>
          <button
            className={`${styles.ctrlBtn} ${styles.btnPrimary}`}
            onClick={togglePaused}
            title="Play or pause the autonomous defense loop"
          >
            {isPaused ? '▶ Run Defense' : '⏸ Pause Defense'}
          </button>

          {onStepOnce && (
            <button
              className={styles.ctrlBtn}
              onClick={onStepOnce}
              title="Execute exactly one step forward so you can inspect the result"
            >
              ⏭ Step Forward
            </button>
          )}

          {onRestart && (
            <button
              className={styles.ctrlBtn}
              onClick={onRestart}
              title="Reset the network back to initial state"
            >
              🔄 Reset Network
            </button>
          )}
        </div>
      </div>

      {/* CURRENT SITUATION SUMMARY */}
      {latestEntry ? (
        <div className={styles.stepHeadline}>
          <div className={styles.headlineTop}>
            <span>CURRENT STATUS // STEP {latestEntry.step}</span>
            <span>REWARD: {latestEntry.reward >= 0 ? `+${latestEntry.reward.toFixed(2)}` : latestEntry.reward.toFixed(2)}</span>
          </div>
          <div className={styles.headlineText}>
            <strong>Action:</strong> {latestEntry.action.tool} &nbsp;—&nbsp; {latestEntry.action.reasoning}
          </div>
        </div>
      ) : (
        <div className={styles.stepHeadline}>
          <div className={styles.headlineTop}>
            <span>READY TO ENGAGE // {scenario?.name || 'Simulation'}</span>
          </div>
          <div className={styles.headlineText}>
            Click <strong>[▶ Run Defense]</strong> to watch the AI defend autonomously, or <strong>[⏭ Step Forward]</strong> to step through one decision at a time.
          </div>
        </div>
      )}

      {/* TIMELINE OF AI DECISIONS */}
      <div className={styles.logArea} ref={scrollRef}>
        {agentLog.length === 0 ? (
          <div className={styles.emptyState}>
            <div style={{ fontSize: 24 }}>🛡️</div>
            <div style={{ fontWeight: 600, color: '#f0f6fc' }}>Autonomous Defender Ready</div>
            <div style={{ fontSize: 12, maxWidth: 280 }}>
              The AI agent is waiting for you to start the defense or step through the attack scenario.
            </div>
          </div>
        ) : (
          agentLog.map((entry, idx) => {
            const rewardFmt = entry.reward >= 0 ? `+${entry.reward.toFixed(2)}` : entry.reward.toFixed(2);
            return (
              <div key={idx} className={styles.stepCard}>
                <div className={styles.stepCardHeader}>
                  <span className={styles.stepBadge}>STEP {entry.step}</span>
                  <span className={`${styles.rewardBadge} ${entry.reward >= 0 ? styles.rewardPos : styles.rewardNeg}`}>
                    {rewardFmt} Score
                  </span>
                </div>

                <div className={styles.actionLine}>
                  {entry.action.tool}({JSON.stringify(entry.action.args)})
                </div>

                <div className={styles.reasoningText}>
                  {entry.action.reasoning}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
