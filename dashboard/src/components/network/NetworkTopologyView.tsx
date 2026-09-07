'use client';

// clean 2D enterprise architecture topology diagram (Linear / Vercel design language)
// organized into clear operational zones with readable status badges and host inspector

import React from 'react';
import { useAppStore } from '@/store';
import type { NetworkNode, NodeStatus } from '@/engine/types';
import styles from './NetworkTopologyView.module.css';

const NODE_ICONS: Record<string, string> = {
  firewall: '🛡️',
  domain_controller: '🏛️',
  web_server: '🌐',
  mail_server: '✉️',
  app_server: '⚙️',
  database: '🗄️',
  backup_server: '💾',
  workstation: '💻',
  honeypot: '🍯',
};

export function NetworkTopologyView() {
  const topology = useAppStore((s) => s.topology);
  const selectedNodeId = useAppStore((s) => s.selectedNodeId);
  const setSelectedNodeId = useAppStore((s) => s.setSelectedNodeId);
  const alerts = useAppStore((s) => s.alerts);

  const selectedNode = topology.find((n) => n.nodeId === selectedNodeId) || null;

  // active attacked node IDs
  const activeAttacks = alerts
    .filter((a) => (a.status === 'new' || a.status === 'investigating') && !a.isFalsePositive)
    .map((a) => a.targetNodeId);

  // partition nodes into enterprise zones
  const dmzNodes = topology.filter((n) => ['fw-01', 'web-01', 'mail-01'].includes(n.nodeId));
  const coreNodes = topology.filter((n) => ['dc-01', 'app-01'].includes(n.nodeId));
  const vaultNodes = topology.filter((n) => ['db-01', 'backup-01'].includes(n.nodeId));
  const endpointNodes = topology.filter((n) => n.type === 'workstation');
  const decoyNodes = topology.filter((n) => n.type === 'honeypot');

  const getStatusBadge = (status: NodeStatus, isAttacked: boolean) => {
    if (status === 'isolated') {
      return <span className={`${styles.statusBadge} ${styles.statusIsolated}`}>🔒 Isolated by AI</span>;
    }
    if (status === 'encrypted') {
      return <span className={`${styles.statusBadge} ${styles.statusEncrypted}`}>💀 Encrypted</span>;
    }
    if (status === 'compromised' || isAttacked) {
      return <span className={`${styles.statusBadge} ${styles.statusCompromised}`}>⚠️ Under Attack</span>;
    }
    return <span className={`${styles.statusBadge} ${styles.statusHealthy}`}>● Normal</span>;
  };

  const renderNodeCard = (node: NetworkNode) => {
    const isSelected = selectedNodeId === node.nodeId;
    const isAttacked = activeAttacks.includes(node.nodeId);
    const isCompromised = node.status === 'compromised' || isAttacked;
    const isIsolated = node.status === 'isolated';

    let cardClass = styles.nodeCard;
    if (isSelected) cardClass += ` ${styles.nodeCardSelected}`;
    if (isCompromised) cardClass += ` ${styles.nodeCardCompromised}`;
    if (isIsolated) cardClass += ` ${styles.nodeCardIsolated}`;

    return (
      <div
        key={node.nodeId}
        className={cardClass}
        onClick={() => setSelectedNodeId(isSelected ? null : node.nodeId)}
      >
        <div className={styles.nodeTop}>
          <div className={styles.nodeTitle}>
            <span>{NODE_ICONS[node.type] || '🖥️'}</span>
            <span>{node.hostname}</span>
          </div>
          {getStatusBadge(node.status, isAttacked)}
        </div>

        <div className={styles.nodeSub}>
          <span>IP: {node.ip}</span>
          <span style={{ color: '#58a6ff' }}>[{node.nodeId}]</span>
        </div>
      </div>
    );
  };

  return (
    <div className={styles.container}>
      <div className={styles.topBarInfo}>
        <span style={{ fontWeight: 600, color: '#f0f6fc', letterSpacing: '0.04em' }}>
          ENTERPRISE ARCHITECTURE TOPOLOGY (12 HOSTS)
        </span>
        <div className={styles.zoneLegend}>
          <div className={styles.legendItem}>
            <span className={styles.legendDot} style={{ background: '#3fb950' }} />
            <span>Healthy Host</span>
          </div>
          <div className={styles.legendItem}>
            <span className={styles.legendDot} style={{ background: '#f85149' }} />
            <span>Active Intrusion</span>
          </div>
          <div className={styles.legendItem}>
            <span className={styles.legendDot} style={{ background: '#58a6ff' }} />
            <span>Isolated by AI Agent</span>
          </div>
        </div>
      </div>

      <div className={styles.diagramArea}>
        <div className={styles.zoneGrid}>
          {/* ZONE 1: DMZ */}
          <div className={styles.zoneBox}>
            <div className={styles.zoneHeader}>
              <span>Perimeter & DMZ</span>
              <span style={{ color: '#58a6ff', fontSize: 10 }}>Public Facing</span>
            </div>
            <div className={styles.nodeList}>
              {dmzNodes.map(renderNodeCard)}
            </div>
          </div>

          {/* ZONE 2: CORE SERVICES */}
          <div className={styles.zoneBox}>
            <div className={styles.zoneHeader}>
              <span>Core Enterprise LAN</span>
              <span style={{ color: '#58a6ff', fontSize: 10 }}>Identity & Apps</span>
            </div>
            <div className={styles.nodeList}>
              {coreNodes.map(renderNodeCard)}
            </div>
          </div>

          {/* ZONE 3: DATA VAULT */}
          <div className={styles.zoneBox}>
            <div className={styles.zoneHeader}>
              <span>Secure Data Vault</span>
              <span style={{ color: '#d29922', fontSize: 10 }}>High Value Target</span>
            </div>
            <div className={styles.nodeList}>
              {vaultNodes.map(renderNodeCard)}
            </div>
          </div>
        </div>

        <div className={styles.zoneGrid} style={{ gridTemplateColumns: '2fr 1fr' }}>
          {/* ZONE 4: WORKSTATIONS */}
          <div className={styles.zoneBox}>
            <div className={styles.zoneHeader}>
              <span>User Endpoint Fleet</span>
              <span style={{ color: '#8b949e', fontSize: 10 }}>Initial Phishing Vectors</span>
            </div>
            <div className={styles.endpointList}>
              {endpointNodes.map(renderNodeCard)}
            </div>
          </div>

          {/* ZONE 5: HONEYPOT */}
          <div className={styles.zoneBox}>
            <div className={styles.zoneHeader}>
              <span>Deception Net</span>
              <span style={{ color: '#d29922', fontSize: 10 }}>Adversary Trap</span>
            </div>
            <div className={styles.nodeList}>
              {decoyNodes.map(renderNodeCard)}
            </div>
          </div>
        </div>
      </div>

      {/* NODE INSPECTION DRAWER */}
      {selectedNode && (
        <aside className={styles.inspectorDrawer}>
          <div className={styles.inspectorHeader}>
            <span className={styles.inspectorTitle}>
              {NODE_ICONS[selectedNode.type]} {selectedNode.hostname}
            </span>
            <button className={styles.inspectorClose} onClick={() => setSelectedNodeId(null)}>
              ✕
            </button>
          </div>

          <div className={styles.inspectorRow}>
            <span className={styles.inspectorKey}>Role & Type</span>
            <span className={styles.inspectorVal}>{selectedNode.type.replace('_', ' ').toUpperCase()}</span>
          </div>

          <div className={styles.inspectorRow}>
            <span className={styles.inspectorKey}>IP Address</span>
            <span className={styles.inspectorVal}>{selectedNode.ip}</span>
          </div>

          <div className={styles.inspectorRow}>
            <span className={styles.inspectorKey}>Operating System</span>
            <span className={styles.inspectorVal}>{selectedNode.os}</span>
          </div>

          <div className={styles.inspectorRow}>
            <span className={styles.inspectorKey}>Current Status</span>
            <span className={styles.inspectorVal} style={{ fontWeight: 600 }}>
              {selectedNode.status.toUpperCase()}
            </span>
          </div>

          <div className={styles.inspectorRow}>
            <span className={styles.inspectorKey}>Active Listening Services</span>
            <span className={styles.inspectorVal}>
              {selectedNode.services.join(', ') || 'None'}
            </span>
          </div>

          <div className={styles.inspectorRow}>
            <span className={styles.inspectorKey}>Open Network Ports</span>
            <span className={styles.inspectorVal}>
              {selectedNode.openPorts.map((p) => `:${p}`).join(' ') || 'None'}
            </span>
          </div>

          {selectedNode.vulnerabilities.length > 0 && (
            <div className={styles.inspectorRow}>
              <span className={styles.inspectorKey} style={{ color: '#f85149' }}>Known Vulnerabilities</span>
              <span className={styles.inspectorVal} style={{ color: '#f85149' }}>
                {selectedNode.vulnerabilities.join(', ')}
              </span>
            </div>
          )}

          <div style={{ marginTop: 'auto', paddingTop: 10 }}>
            <button
              style={{
                width: '100%',
                padding: '8px',
                background: '#21262d',
                border: '1px solid #30363d',
                borderRadius: '4px',
                color: '#c9d1d9',
                cursor: 'pointer',
                fontFamily: 'var(--font-ui)',
                fontSize: '11px',
              }}
              onClick={() => setSelectedNodeId(null)}
            >
              Close Inspector
            </button>
          </div>
        </aside>
      )}
    </div>
  );
}
