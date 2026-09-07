'use client';

// threat intelligence feed sidebar
// shows real threat actor profile, CVEs, and IOC tracker for the active scenario
import React, { useState } from 'react';
import { useAppStore } from '@/store';
import { getThreatIntel } from '@/engine/threat-intel';
import type { IOC } from '@/engine/threat-intel';
import styles from './ThreatIntelFeed.module.css';

type Tab = 'actor' | 'cves' | 'iocs';

export function ThreatIntelFeed() {
  const scenario = useAppStore((s) => s.scenario);
  const agentLog = useAppStore((s) => s.agentLog);
  const [activeTab, setActiveTab] = useState<Tab>('actor');

  const intel = scenario ? getThreatIntel(scenario.id) : null;

  // derive neutralized IOCs from agent log actions
  const neutralizedSet = new Set<string>();
  agentLog.forEach((entry) => {
    const tool = entry.action.tool;
    const args = entry.action.args;
    if (tool === 'block_ip' && args.ip) neutralizedSet.add(args.ip);
    if (tool === 'isolate_host' && args.host) neutralizedSet.add(args.host);
    if (entry.result.success && (tool === 'run_forensics' || tool === 'deploy_patch')) {
      // mark all pending IOCs as checked after successful forensics
      if (args.host) neutralizedSet.add(`patch:${args.host}`);
    }
  });

  if (!intel) {
    return (
      <div className={styles.container}>
        <div className={styles.header}>
          <span className={styles.headerLabel}>Threat Intelligence</span>
          <span className={styles.liveTag}>SIGINT</span>
        </div>
        <div className={styles.empty}>
          <span className={styles.emptyIcon}>⬡</span>
          <span>No scenario active. Launch an attack campaign to see threat intelligence.</span>
        </div>
      </div>
    );
  }

  const { actor, cves, iocs } = intel;

  // calculate IOC status — neutralize IPs and domains that the agent acted on
  const resolvedIocs: (IOC & { resolvedStatus: 'active' | 'neutralized' | 'pending' })[] = iocs.map((ioc) => {
    const isNeutralized =
      (ioc.type === 'IP' && neutralizedSet.has(ioc.value)) ||
      ioc.status === 'neutralized';
    return {
      ...ioc,
      resolvedStatus: isNeutralized ? 'neutralized' : ioc.status,
    };
  });

  const activeIocsCount = resolvedIocs.filter((i) => i.resolvedStatus === 'active').length;
  const neutralizedCount = resolvedIocs.filter((i) => i.resolvedStatus === 'neutralized').length;

  return (
    <div className={styles.container}>
      {/* header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <span className={styles.headerLabel}>Threat Intelligence</span>
          <span className={styles.liveTag}>SIGINT FEED</span>
        </div>
        <div className={styles.iocSummary}>
          <span style={{ color: 'var(--red)' }}>{activeIocsCount} active</span>
          <span style={{ color: 'var(--text-3)' }}>·</span>
          <span style={{ color: 'var(--green)' }}>{neutralizedCount} cleared</span>
        </div>
      </div>

      {/* tabs */}
      <div className={styles.tabs}>
        <button
          className={`${styles.tab} ${activeTab === 'actor' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('actor')}
        >
          Threat Actor
        </button>
        <button
          className={`${styles.tab} ${activeTab === 'cves' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('cves')}
        >
          CVEs ({cves.length})
        </button>
        <button
          className={`${styles.tab} ${activeTab === 'iocs' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('iocs')}
        >
          IOC Tracker ({iocs.length})
        </button>
      </div>

      {/* tab content */}
      <div className={styles.body}>
        {activeTab === 'actor' && (
          <div className={styles.actorPanel}>
            {/* actor identity */}
            <div className={styles.actorCard}>
              <div className={styles.actorBadge}>
                <span className={styles.threatDot} />
                CONFIRMED THREAT ACTOR
              </div>
              <div className={styles.actorCodename}>{actor.codename}</div>
              <div className={styles.actorAlias}>{actor.alias}</div>
            </div>

            <div className={styles.fieldList}>
              <Field label="Origin" value={actor.origin} />
              <Field label="Motivation" value={actor.motivation} />
              <Field label="Target Sectors" value={actor.sectors.join(', ')} />
              {actor.mitreGroups.length > 0 && (
                <Field label="MITRE Groups" value={actor.mitreGroups.join(', ')} mono />
              )}
            </div>

            <div className={styles.toolSection}>
              <div className={styles.toolLabel}>Known Tooling & Malware</div>
              <div className={styles.toolChips}>
                {actor.knownTools.map((tool) => (
                  <span key={tool} className={styles.toolChip}>{tool}</span>
                ))}
              </div>
            </div>

            <div className={styles.activityBlock}>
              <div className={styles.activityLabel}>Current Campaign Intelligence</div>
              <p className={styles.activityText}>{actor.activity}</p>
            </div>
          </div>
        )}

        {activeTab === 'cves' && (
          <div className={styles.cvePanel}>
            {cves.map((cve) => (
              <div key={cve.cveId} className={`${styles.cveCard} ${cve.active ? styles.cveActive : ''}`}>
                <div className={styles.cveTopRow}>
                  <div className={styles.cveMeta}>
                    <span className={styles.cveId}>{cve.cveId}</span>
                    {cve.active && <span className={styles.activeExploitBadge}>⚡ IN USE</span>}
                  </div>
                  <div className={styles.cvssRow}>
                    <span className={styles.cvssLabel}>CVSS</span>
                    <span
                      className={styles.cvssScore}
                      style={{
                        color: cve.cvss >= 9 ? 'var(--red)' : cve.cvss >= 7 ? 'var(--orange)' : 'var(--amber)',
                      }}
                    >
                      {cve.cvss.toFixed(1)}
                    </span>
                  </div>
                </div>

                <div className={styles.cveComponent}>{cve.component}</div>
                <p className={styles.cveDesc}>{cve.description}</p>

                <div className={styles.exploitStatus}>
                  <span
                    className={styles.exploitBadge}
                    style={{
                      color: cve.exploitStatus === 'Actively Exploited' ? 'var(--red)'
                        : cve.exploitStatus === 'PoC Available' ? 'var(--orange)'
                        : 'var(--text-2)',
                      borderColor: cve.exploitStatus === 'Actively Exploited' ? 'rgba(239,68,68,0.25)'
                        : cve.exploitStatus === 'PoC Available' ? 'rgba(249,115,22,0.25)'
                        : 'var(--border)',
                    }}
                  >
                    {cve.exploitStatus}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}

        {activeTab === 'iocs' && (
          <div className={styles.iocPanel}>
            <div className={styles.iocProgress}>
              <span className={styles.iocProgressLabel}>
                {neutralizedCount} / {resolvedIocs.length} IOCs Cleared
              </span>
              <div className={styles.iocProgressTrack}>
                <div
                  className={styles.iocProgressFill}
                  style={{ width: `${resolvedIocs.length > 0 ? (neutralizedCount / resolvedIocs.length) * 100 : 0}%` }}
                />
              </div>
            </div>

            <div className={styles.iocList}>
              {resolvedIocs.map((ioc, i) => (
                <div
                  key={i}
                  className={`${styles.iocRow} ${ioc.resolvedStatus === 'neutralized' ? styles.iocNeutralized : ioc.resolvedStatus === 'active' ? styles.iocActiveRow : ''}`}
                >
                  <div className={styles.iocIndicator}>
                    {ioc.resolvedStatus === 'neutralized' ? (
                      <span className={styles.iocCheckmark}>✓</span>
                    ) : ioc.resolvedStatus === 'active' ? (
                      <span className={styles.iocAlert}>!</span>
                    ) : (
                      <span className={styles.iocPending}>○</span>
                    )}
                  </div>
                  <div className={styles.iocContent}>
                    <div className={styles.iocType}>{ioc.type}</div>
                    <div className={styles.iocValue}>{ioc.value}</div>
                    <div className={styles.iocDesc}>{ioc.description}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Field({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className={styles.field}>
      <span className={styles.fieldLabel}>{label}</span>
      <span className={styles.fieldValue} style={{ fontFamily: mono ? 'var(--font-mono)' : undefined }}>
        {value}
      </span>
    </div>
  );
}
