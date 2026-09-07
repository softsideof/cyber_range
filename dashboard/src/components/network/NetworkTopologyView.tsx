'use client';

// enterprise network topology — zone-based architecture diagram
// nodes show animated attack pulse when under threat
import React from 'react';
import { useAppStore } from '@/store';
import type { NetworkNode, NodeStatus } from '@/engine/types';
import styles from './NetworkTopologyView.module.css';

// SVG role icons — clean vector, no emojis
const RoleIcon = ({ type, size = 14 }: { type: string; size?: number }) => {
  const color = 'currentColor';
  switch (type) {
    case 'firewall':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        </svg>
      );
    case 'domain_controller':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
          <rect x="2" y="3" width="20" height="14" rx="2" />
          <path d="M8 21h8M12 17v4" />
        </svg>
      );
    case 'web_server':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
        </svg>
      );
    case 'mail_server':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
          <rect x="2" y="4" width="20" height="16" rx="2" />
          <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7" />
        </svg>
      );
    case 'app_server':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
          <rect x="2" y="2" width="20" height="8" rx="2" />
          <rect x="2" y="14" width="20" height="8" rx="2" />
          <path d="M6 6h.01M6 18h.01" />
        </svg>
      );
    case 'database':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
          <ellipse cx="12" cy="5" rx="9" ry="3" />
          <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
          <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
        </svg>
      );
    case 'workstation':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
          <rect x="2" y="3" width="20" height="14" rx="2" />
          <path d="M8 21h8M12 17v4" />
          <path d="M7 8h10M7 12h6" />
        </svg>
      );
    case 'backup_server':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
      );
    case 'honeypot':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
          <path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z" />
          <path d="M12 8v4l3 3" />
        </svg>
      );
    default:
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
          <rect x="2" y="3" width="20" height="14" rx="2" />
        </svg>
      );
  }
};

const STATUS_LABEL: Record<NodeStatus, string> = {
  healthy:      'Healthy',
  compromised:  'Compromised',
  isolated:     'Isolated',
  encrypted:    'Encrypted',
  offline:      'Offline',
  patched:      'Patched',
};

export function NetworkTopologyView() {
  const topology        = useAppStore((s) => s.topology);
  const selectedNodeId  = useAppStore((s) => s.selectedNodeId);
  const setSelectedNodeId = useAppStore((s) => s.setSelectedNodeId);
  const alerts          = useAppStore((s) => s.alerts);

  const selectedNode = topology.find((n) => n.nodeId === selectedNodeId) || null;

  // nodes with active unresolved attacks
  const attackedIds = new Set(
    alerts
      .filter((a) => (a.status === 'new' || a.status === 'investigating') && !a.isFalsePositive)
      .map((a) => a.targetNodeId),
  );

  // zone partitions
  const perimeter = topology.filter((n) => ['fw-01', 'web-01', 'mail-01'].includes(n.nodeId));
  const core      = topology.filter((n) => ['dc-01', 'app-01'].includes(n.nodeId));
  const vault     = topology.filter((n) => ['db-01', 'backup-01'].includes(n.nodeId));
  const endpoints = topology.filter((n) => n.type === 'workstation');
  const deception = topology.filter((n) => n.type === 'honeypot');

  // live counts
  const compromisedCount = topology.filter((n) => n.status === 'compromised').length;
  const isolatedCount    = topology.filter((n) => n.status === 'isolated').length;
  const healthyCount     = topology.filter((n) => n.status === 'healthy').length;

  const renderNode = (node: NetworkNode) => {
    const isSelected    = node.nodeId === selectedNodeId;
    const isAttacked    = attackedIds.has(node.nodeId);
    const isCompromised = node.status === 'compromised' || isAttacked;
    const isIsolated    = node.status === 'isolated';
    const isEncrypted   = node.status === 'encrypted';

    let cardMod = '';
    if (isSelected)    cardMod = styles.nodeSelected;
    else if (isEncrypted)  cardMod = styles.nodeEncrypted;
    else if (isCompromised) cardMod = styles.nodeCompromised;
    else if (isIsolated) cardMod = styles.nodeIsolated;

    return (
      <button
        key={node.nodeId}
        className={`${styles.nodeCard} ${cardMod}`}
        onClick={() => setSelectedNodeId(isSelected ? null : node.nodeId)}
        title={`${node.hostname} (${node.ip}) — click to inspect`}
      >
        <span className={styles.nodeIcon}>
          <RoleIcon type={node.type} size={13} />
        </span>
        <div className={styles.nodeInfo}>
          <span className={styles.nodeHostname}>{node.hostname}</span>
          <span className={styles.nodeIp}>{node.ip}</span>
        </div>
        <span className={`${styles.nodeStatus} ${getStatusMod(node.status, isAttacked)}`}>
          {getStatusDot(node.status, isAttacked)}
          {isAttacked && node.status !== 'isolated' ? 'Under Attack' : STATUS_LABEL[node.status]}
        </span>
      </button>
    );
  };

  const getStatusMod = (status: NodeStatus, attacked: boolean) => {
    if (status === 'isolated')   return styles.statusIsolated;
    if (status === 'encrypted')  return styles.statusEncrypted;
    if (status === 'compromised' || attacked) return styles.statusCompromised;
    if (status === 'patched')    return styles.statusPatched;
    return styles.statusHealthy;
  };

  const getStatusDot = (status: NodeStatus, attacked: boolean) => {
    let color = 'var(--green)';
    if (status === 'isolated') color = 'var(--purple)';
    else if (status === 'encrypted') color = 'var(--red)';
    else if (status === 'compromised' || attacked) color = 'var(--red)';
    else if (status === 'patched') color = 'var(--cyan)';
    return <span className={styles.dot} style={{ background: color }} />;
  };

  const renderZone = (
    label: string,
    sublabel: string,
    nodes: NetworkNode[],
    accentColor: string,
    isRow = false,
  ) => (
    <div className={styles.zone}>
      <div className={styles.zoneHeader} style={{ borderLeftColor: accentColor }}>
        <span className={styles.zoneLabel}>{label}</span>
        <span className={styles.zoneSublabel}>{sublabel}</span>
      </div>
      <div className={isRow ? styles.nodeRow : styles.nodeList}>
        {nodes.map(renderNode)}
      </div>
    </div>
  );

  return (
    <div className={styles.container}>
      {/* header — live host status summary */}
      <div className={styles.header}>
        <span className={styles.title}>Enterprise Network — 12 Hosts</span>
        <div className={styles.hostSummary}>
          <span className={styles.hostCount} style={{ color: 'var(--green)' }}>
            {healthyCount} healthy
          </span>
          {compromisedCount > 0 && (
            <span className={styles.hostCount} style={{ color: 'var(--red)' }}>
              {compromisedCount} compromised
            </span>
          )}
          {isolatedCount > 0 && (
            <span className={styles.hostCount} style={{ color: 'var(--purple)' }}>
              {isolatedCount} isolated
            </span>
          )}
        </div>
      </div>

      {/* zone diagram */}
      <div className={styles.diagram}>
        {/* row 1: perimeter + core + vault */}
        <div className={styles.row3}>
          {renderZone('Perimeter & DMZ', 'Public Facing', perimeter, 'var(--blue)')}
          {renderZone('Core Enterprise', 'Identity & Apps', core, 'var(--amber)')}
          {renderZone('Secure Vault', 'High Value Target', vault, 'var(--red)')}
        </div>
        {/* row 2: endpoints + deception */}
        <div className={styles.row2}>
          {renderZone('User Endpoints', 'Phishing Vectors', endpoints, 'var(--text-3)', true)}
          {renderZone('Deception Net', 'Honeypot Trap', deception, 'var(--purple)')}
        </div>
      </div>

      {/* inspector drawer */}
      {selectedNode && (
        <aside className={`${styles.drawer} animate-fade-in`}>
          <div className={styles.drawerHeader}>
            <div className={styles.drawerTitle}>
              <span className={styles.drawerIcon}><RoleIcon type={selectedNode.type} size={15} /></span>
              <span>{selectedNode.hostname}</span>
            </div>
            <button className={styles.drawerClose} onClick={() => setSelectedNodeId(null)}>✕</button>
          </div>

          <div className={styles.drawerBody}>
            <Row label="Node ID"   value={selectedNode.nodeId} mono />
            <Row label="IP Address" value={selectedNode.ip} mono />
            <Row label="Role"      value={selectedNode.type.replace(/_/g, ' ').toUpperCase()} />
            <Row label="OS"        value={selectedNode.os} />
            <Row
              label="Status"
              value={selectedNode.status.toUpperCase()}
              color={
                selectedNode.status === 'healthy' ? 'var(--green)' :
                selectedNode.status === 'isolated' ? 'var(--purple)' :
                selectedNode.status === 'compromised' ? 'var(--red)' : 'var(--amber)'
              }
            />
            <Row label="Services"  value={selectedNode.services.join(', ') || 'none'} />
            <Row label="Ports"     value={selectedNode.openPorts.map((p) => `:${p}`).join(' ') || 'none'} mono />
            {selectedNode.vulnerabilities.length > 0 && (
              <Row
                label="CVEs"
                value={selectedNode.vulnerabilities.join(' · ')}
                color="var(--red)"
                mono
              />
            )}
            <Row label="Critical Asset" value={selectedNode.isCritical ? 'YES' : 'NO'} color={selectedNode.isCritical ? 'var(--amber)' : 'var(--text-2)'} />
          </div>

          <button className={styles.drawerCloseBtn} onClick={() => setSelectedNodeId(null)}>
            Close Inspector
          </button>
        </aside>
      )}
    </div>
  );
}

function Row({ label, value, mono, color }: { label: string; value: string; mono?: boolean; color?: string }) {
  return (
    <div className={styles.drawerRow}>
      <span className={styles.drawerKey}>{label}</span>
      <span
        className={styles.drawerVal}
        style={{
          fontFamily: mono ? 'var(--font-mono)' : undefined,
          color: color,
        }}
      >
        {value}
      </span>
    </div>
  );
}
