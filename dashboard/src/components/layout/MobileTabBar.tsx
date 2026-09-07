'use client';

// bottom navigation tab bar for mobile viewports (< 1024px)
import React from 'react';
import { useAppStore } from '@/store';
import styles from './MobileTabBar.module.css';

export function MobileTabBar() {
  const activeTab    = useAppStore((s) => s.activeTab);
  const setActiveTab = useAppStore((s) => s.setActiveTab);
  const alerts       = useAppStore((s) => s.alerts);

  const unhandled = alerts.filter((a) => a.status === 'new' || a.status === 'investigating').length;

  return (
    <nav className={styles.mobileTabBar}>
      <button
        className={`${styles.tabItem} ${activeTab === 'network' ? styles.tabItemActive : ''}`}
        onClick={() => setActiveTab('network')}
      >
        <span className={styles.tabIcon}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/>
          </svg>
        </span>
        <span>Network</span>
      </button>

      <button
        className={`${styles.tabItem} ${activeTab === 'alerts' ? styles.tabItemActive : ''}`}
        onClick={() => setActiveTab('alerts')}
      >
        <span className={styles.tabIcon}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
        </span>
        <span>Alerts</span>
        {unhandled > 0 && <span className={styles.tabBadge}>{unhandled}</span>}
      </button>

      <button
        className={`${styles.tabItem} ${activeTab === 'intel' ? styles.tabItemActive : ''}`}
        onClick={() => setActiveTab('intel')}
      >
        <span className={styles.tabIcon}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
        </span>
        <span>Intel</span>
      </button>

      <button
        className={`${styles.tabItem} ${activeTab === 'agent' ? styles.tabItemActive : ''}`}
        onClick={() => setActiveTab('agent')}
      >
        <span className={styles.tabIcon}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"/><path d="M9 9l3 3 3-3m-3 3v3"/>
          </svg>
        </span>
        <span>Defense</span>
      </button>

      <button
        className={`${styles.tabItem} ${activeTab === 'score' ? styles.tabItemActive : ''}`}
        onClick={() => setActiveTab('score')}
      >
        <span className={styles.tabIcon}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
          </svg>
        </span>
        <span>Results</span>
      </button>
    </nav>
  );
}
