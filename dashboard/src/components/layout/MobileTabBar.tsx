'use client';

// bottom navigation tab bar for mobile viewports (< 1024px)

import React from 'react';
import { useAppStore } from '@/store';
import styles from './MobileTabBar.module.css';

export function MobileTabBar() {
  const activeTab = useAppStore((s) => s.activeTab);
  const setActiveTab = useAppStore((s) => s.setActiveTab);
  const alerts = useAppStore((s) => s.alerts);

  const unhandledAlerts = alerts.filter((a) => a.status === 'new' || a.status === 'investigating').length;

  return (
    <nav className={styles.mobileTabBar}>
      <button
        className={`${styles.tabItem} ${activeTab === 'network' ? styles.tabItemActive : ''}`}
        onClick={() => setActiveTab('network')}
      >
        <span>🌐</span>
        <span>Network</span>
      </button>

      <button
        className={`${styles.tabItem} ${activeTab === 'alerts' ? styles.tabItemActive : ''}`}
        onClick={() => setActiveTab('alerts')}
      >
        <span>🚨</span>
        <span>Alerts</span>
        {unhandledAlerts > 0 && <span className={styles.tabBadge}>{unhandledAlerts}</span>}
      </button>

      <button
        className={`${styles.tabItem} ${activeTab === 'agent' ? styles.tabItemActive : ''}`}
        onClick={() => setActiveTab('agent')}
      >
        <span>🤖</span>
        <span>Agent</span>
      </button>

      <button
        className={`${styles.tabItem} ${activeTab === 'mitre' ? styles.tabItemActive : ''}`}
        onClick={() => setActiveTab('mitre')}
      >
        <span>🛡️</span>
        <span>MITRE</span>
      </button>

      <button
        className={`${styles.tabItem} ${activeTab === 'score' ? styles.tabItemActive : ''}`}
        onClick={() => setActiveTab('score')}
      >
        <span>📊</span>
        <span>Results</span>
      </button>
    </nav>
  );
}
