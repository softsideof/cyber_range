'use client';

// main dashboard page — network map + threat intelligence + defense log
import React, { useState, useEffect } from 'react';
import { useAppStore } from '@/store';
import { useWorker } from '@/hooks/useWorker';
import { useKeyboard } from '@/hooks/useKeyboard';
import { TopBar } from '@/components/layout/TopBar';
import { StatusBar } from '@/components/layout/StatusBar';
import { MobileTabBar } from '@/components/layout/MobileTabBar';
import { MetricsBar } from '@/components/panels/MetricsBar';
import { NetworkTopologyView } from '@/components/network/NetworkTopologyView';
import { AlertQueue } from '@/components/panels/AlertQueue';
import { AgentLog } from '@/components/panels/AgentLog';
import { ThreatIntelFeed } from '@/components/panels/ThreatIntelFeed';
import { MitreHeatmap } from '@/components/panels/MitreHeatmap';
import { ScoreCard } from '@/components/panels/ScoreCard';
import { ScenarioBriefing } from '@/components/panels/ScenarioBriefing';
import { ArchitectureView } from '@/components/panels/ArchitectureView';
import { AttackBuilder } from '@/components/attack-builder/AttackBuilder';
import { ShortcutsModal } from '@/components/panels/ShortcutsModal';
import styles from './page.module.css';

export default function DashboardPage() {
  const { launch, launchCustom, restart, stepOnce } = useWorker();
  useKeyboard(launch, stepOnce, restart);

  const mode       = useAppStore((s) => s.mode);
  const activeTab  = useAppStore((s) => s.activeTab);
  const showBriefing = useAppStore((s) => s.showBriefing);
  const setShowBriefing    = useAppStore((s) => s.setShowBriefing);
  const setShowAttackBuilder = useAppStore((s) => s.setShowAttackBuilder);
  const nextScenario = useAppStore((s) => s.nextScenario);

  // right column bottom-section tab
  const [rightTab, setRightTab] = useState<'agent' | 'alerts' | 'mitre'>('agent');
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 1024);
    check();
    window.addEventListener('resize', check);
    return () => window.removeEventListener('resize', check);
  }, []);

  const handleNextScenario = () => {
    const next = nextScenario();
    launch(next.id);
  };

  return (
    <div className={styles.shell}>
      <TopBar onSelectScenario={launch} />
      <MetricsBar />

      <main className={styles.workspace}>
        {isMobile ? (
          // mobile: single active panel
          <div className={styles.mobileOnlyPanel}>
            {activeTab === 'network' && (
              mode === 'complete' ? (
                <ScoreCard
                  onRestart={restart}
                  onOpenBuilder={() => setShowAttackBuilder(true)}
                  onNext={handleNextScenario}
                />
              ) : (
                <NetworkTopologyView />
              )
            )}
            {activeTab === 'alerts'  && <AlertQueue />}
            {activeTab === 'agent'   && <AgentLog onStepOnce={stepOnce} onRestart={restart} />}
            {activeTab === 'mitre'   && <MitreHeatmap />}
            {activeTab === 'intel'   && <ThreatIntelFeed />}
            {activeTab === 'score'   && (
              <ScoreCard
                onRestart={restart}
                onOpenBuilder={() => setShowAttackBuilder(true)}
                onNext={handleNextScenario}
              />
            )}
          </div>
        ) : (
          // desktop: 3-column layout
          <>
            {/* LEFT — network map (55%) */}
            <div className={styles.mainVisual}>
              {mode === 'complete' ? (
                <ScoreCard
                  onRestart={restart}
                  onOpenBuilder={() => setShowAttackBuilder(true)}
                  onNext={handleNextScenario}
                />
              ) : (
                <NetworkTopologyView />
              )}
            </div>

            {/* RIGHT COLUMN (45%) — split vertically */}
            <aside className={styles.rightRail}>
              {/* TOP: Threat Intelligence Feed (45%) */}
              <div className={styles.intelSection}>
                <ThreatIntelFeed />
              </div>

              {/* BOTTOM: tabbed — Agent Log / Alerts / MITRE (55%) */}
              <div className={styles.logSection}>
                {/* tab bar */}
                <div className={styles.tabBar}>
                  <button
                    className={`${styles.tabBtn} ${rightTab === 'agent' ? styles.tabActive : ''}`}
                    onClick={() => setRightTab('agent')}
                  >
                    Defense Log
                  </button>
                  <button
                    className={`${styles.tabBtn} ${rightTab === 'alerts' ? styles.tabActive : ''}`}
                    onClick={() => setRightTab('alerts')}
                  >
                    SIEM Alerts
                  </button>
                  <button
                    className={`${styles.tabBtn} ${rightTab === 'mitre' ? styles.tabActive : ''}`}
                    onClick={() => setRightTab('mitre')}
                  >
                    ATT&CK
                  </button>
                </div>

                <div className={styles.tabContent}>
                  {rightTab === 'agent' && <AgentLog onStepOnce={stepOnce} onRestart={restart} />}
                  {rightTab === 'alerts' && <AlertQueue />}
                  {rightTab === 'mitre' && <MitreHeatmap />}
                </div>
              </div>
            </aside>
          </>
        )}
      </main>

      <StatusBar />
      <MobileTabBar />

      {/* overlays */}
      {showBriefing && <ScenarioBriefing onDismiss={() => setShowBriefing(false)} />}
      <ArchitectureView />
      <AttackBuilder onDeployCustom={launchCustom} />
      <ShortcutsModal />
    </div>
  );
}
