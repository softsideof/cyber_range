'use client';

// main dashboard page integrating 3D visual, SIEM telemetry, policy logs, and attack construction

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

  const mode = useAppStore((s) => s.mode);
  const activeTab = useAppStore((s) => s.activeTab);
  const showBriefing = useAppStore((s) => s.showBriefing);
  const setShowBriefing = useAppStore((s) => s.setShowBriefing);
  const setShowAttackBuilder = useAppStore((s) => s.setShowAttackBuilder);
  const nextScenario = useAppStore((s) => s.nextScenario);

  // desktop bottom-right tab state
  const [rightTab, setRightTab] = useState<'agent' | 'mitre'>('agent');
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 1024);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
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
          // Mobile single-panel view
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
            {activeTab === 'alerts' && <AlertQueue />}
            {activeTab === 'agent' && <AgentLog onStepOnce={stepOnce} onRestart={restart} />}
            {activeTab === 'mitre' && <MitreHeatmap />}
            {activeTab === 'score' && (
              <ScoreCard
                onRestart={restart}
                onOpenBuilder={() => setShowAttackBuilder(true)}
                onNext={handleNextScenario}
              />
            )}
          </div>
        ) : (
          // Desktop split layout: 2D Enterprise Network on left, SIEM & AI Story on right
          <>
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

            <aside className={styles.rightRail}>
              <div className={styles.alertSection}>
                <AlertQueue />
              </div>

              <div className={styles.tabbedSection}>
                <div className={styles.tabBar}>
                  <button
                    className={`${styles.tabBtn} ${rightTab === 'agent' ? styles.tabBtnActive : ''}`}
                    onClick={() => setRightTab('agent')}
                  >
                    Incident Story & AI Defense
                  </button>
                  <button
                    className={`${styles.tabBtn} ${rightTab === 'mitre' ? styles.tabBtnActive : ''}`}
                    onClick={() => setRightTab('mitre')}
                  >
                    MITRE ATT&CK Matrix
                  </button>
                </div>

                <div className={styles.tabContent}>
                  {rightTab === 'agent' ? (
                    <AgentLog onStepOnce={stepOnce} onRestart={restart} />
                  ) : (
                    <MitreHeatmap />
                  )}
                </div>
              </div>
            </aside>
          </>
        )}
      </main>

      <StatusBar />
      <MobileTabBar />

      {/* overlays & modals */}
      {showBriefing && <ScenarioBriefing onDismiss={() => setShowBriefing(false)} />}
      <ArchitectureView />
      <AttackBuilder onDeployCustom={launchCustom} />
      <ShortcutsModal />
    </div>
  );
}
