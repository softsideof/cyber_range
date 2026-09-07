'use client';

// tactical 2D planar topology canvas
// renders enterprise sectors, glowing fiber links, hardware icons, and threat breach indicators

import React from 'react';
import { useAppStore } from '@/store';
import { NETWORK_LINKS, NODE_POSITIONS, STATUS_COLORS, getNode } from '@/engine/network';

export function NetworkFallback2D() {
  const topology = useAppStore((s) => s.topology);
  const selectedNodeId = useAppStore((s) => s.selectedNodeId);
  const setSelectedNodeId = useAppStore((s) => s.setSelectedNodeId);
  const alerts = useAppStore((s) => s.alerts);

  const selectedNode = selectedNodeId
    ? getNode({ nodes: topology, blockedIps: new Set(), honeypotActive: false, isolatedNodes: new Set() }, selectedNodeId)
    : null;

  const getCoords = (nodeId: string) => {
    const pos = NODE_POSITIONS[nodeId] || { x: 0, y: 0, z: 0 };
    const x = ((pos.x + 4.5) / 11) * 760 + 40;
    const y = ((pos.z + 3.8) / 7.6) * 440 + 30;
    return { x, y };
  };

  const activeAttackedNodes = alerts
    .filter((a) => (a.status === 'new' || a.status === 'investigating') && !a.isFalsePositive)
    .map((a) => a.targetNodeId);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', background: '#05070c', overflow: 'hidden' }}>
      <svg
        width="100%"
        height="100%"
        viewBox="0 0 840 500"
        preserveAspectRatio="xMidYMid meet"
        style={{ display: 'block' }}
      >
        <defs>
          <radialGradient id="nodeGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#ff2a55" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#ff2a55" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="cyanGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#00f0ff" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#00f0ff" stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* sector demarcations */}
        <rect x="20" y="20" width="800" height="90" fill="rgba(0, 240, 255, 0.02)" stroke="#1a2538" strokeDasharray="3,3" />
        <text x="30" y="38" fill="#00f0ff" fontSize="10" fontFamily="var(--font-mono)" fontWeight="700" letterSpacing="0.08em">
          [SECTOR 01 // PERIMETER GATEWAY & DMZ]
        </text>

        <rect x="20" y="140" width="800" height="150" fill="rgba(58, 134, 255, 0.02)" stroke="#1a2538" strokeDasharray="3,3" />
        <text x="30" y="158" fill="#3a86ff" fontSize="10" fontFamily="var(--font-mono)" fontWeight="700" letterSpacing="0.08em">
          [SECTOR 02 // CORE ARCHITECTURE, DIRECTORY & PRODUCTION DATABASES]
        </text>

        <rect x="20" y="320" width="800" height="155" fill="rgba(255, 183, 3, 0.02)" stroke="#1a2538" strokeDasharray="3,3" />
        <text x="30" y="338" fill="#ffb703" fontSize="10" fontFamily="var(--font-mono)" fontWeight="700" letterSpacing="0.08em">
          [SECTOR 03 // ENDPOINT WORKSTATION FLEET & DECEPTION TRAP]
        </text>

        {/* fiber optic links */}
        {NETWORK_LINKS.map(([src, dst], idx) => {
          const c1 = getCoords(src);
          const c2 = getCoords(dst);
          const isAttacked = activeAttackedNodes.includes(src) || activeAttackedNodes.includes(dst);

          return (
            <line
              key={idx}
              x1={c1.x}
              y1={c1.y}
              x2={c2.x}
              y2={c2.y}
              stroke={isAttacked ? '#ff2a55' : '#1a263c'}
              strokeWidth={isAttacked ? 2.5 : 1.2}
              strokeDasharray={isAttacked ? '6,3' : 'none'}
              style={{ transition: 'all 0.3s ease' }}
            />
          );
        })}

        {/* nodes */}
        {topology.map((node) => {
          const { x, y } = getCoords(node.nodeId);
          const isSelected = selectedNodeId === node.nodeId;
          const isAttacked = activeAttackedNodes.includes(node.nodeId);
          const color = STATUS_COLORS[node.status] || '#06d6a0';

          return (
            <g
              key={node.nodeId}
              onClick={() => setSelectedNodeId(isSelected ? null : node.nodeId)}
              style={{ cursor: 'pointer' }}
            >
              {isAttacked && (
                <circle cx={x} cy={y} r="32" fill="url(#nodeGlow)" />
              )}
              {isSelected && (
                <circle cx={x} cy={y} r="28" fill="url(#cyanGlow)" />
              )}

              {/* isolation shield ring */}
              {node.status === 'isolated' && (
                <circle
                  cx={x}
                  cy={y}
                  r="24"
                  fill="none"
                  stroke="#3a86ff"
                  strokeWidth="2"
                  strokeDasharray="4,4"
                />
              )}

              {/* hardware pedestal chassis */}
              <rect
                x={x - 22}
                y={y - 14}
                width="44"
                height="28"
                rx="2"
                fill="#0a0f1a"
                stroke={isSelected ? 'var(--cyan)' : isAttacked ? 'var(--red)' : '#23304a'}
                strokeWidth={isSelected || isAttacked ? 2 : 1}
              />

              {/* status indicator dot */}
              <circle cx={x - 14} cy={y} r="4" fill={color} />

              {/* node ID */}
              <text
                x={x + 3}
                y={y + 3.5}
                textAnchor="middle"
                fill="#fff"
                fontSize="9"
                fontFamily="var(--font-mono)"
                fontWeight="700"
              >
                {node.nodeId.toUpperCase()}
              </text>

              {/* hostname label */}
              <text
                x={x}
                y={y + 24}
                textAnchor="middle"
                fill="var(--text-2)"
                fontSize="9"
                fontFamily="var(--font-mono)"
              >
                {node.hostname}
              </text>
            </g>
          );
        })}
      </svg>

      {/* selected node inspection detail card */}
      {selectedNode && (
        <div
          style={{
            position: 'absolute',
            bottom: 16,
            left: 16,
            background: 'rgba(10, 14, 23, 0.95)',
            border: '1px solid var(--border-bright)',
            padding: '14px 18px',
            borderRadius: 3,
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            zIndex: 10,
            maxWidth: 320,
            boxShadow: '0 12px 36px rgba(0,0,0,0.8)',
            borderLeft: `3px solid ${STATUS_COLORS[selectedNode.status]}`,
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
            <span style={{ fontWeight: 800, color: '#fff', fontSize: 12 }}>{selectedNode.hostname.toUpperCase()}</span>
            <span style={{ color: STATUS_COLORS[selectedNode.status], fontWeight: 700, fontSize: 10 }}>
              {selectedNode.status.toUpperCase()}
            </span>
          </div>
          <div style={{ color: 'var(--text-2)', fontSize: 10 }}>IP: {selectedNode.ip} • OS: {selectedNode.os}</div>
          <div style={{ color: 'var(--text-2)', fontSize: 10, marginTop: 4 }}>
            SERVICES: {selectedNode.services.join(', ') || 'NONE'}
          </div>
        </div>
      )}
    </div>
  );
}
