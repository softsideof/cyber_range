'use client';

// interactive 2D network map for picking attack target nodes

import React from 'react';
import { NODE_POSITIONS } from '@/engine/network';
import { useAppStore } from '@/store';

interface NodeSelectorProps {
  selectedNodes: string[];
  onToggleNode: (nodeId: string) => void;
}

export function NodeSelector({ selectedNodes, onToggleNode }: NodeSelectorProps) {
  const topology = useAppStore((s) => s.topology);

  // transform 3D coordinates (x: -3 to 5, z: -3 to 3) to 2D SVG canvas (width 400, height 220)
  const mapCoord = (nodeId: string) => {
    const pos = NODE_POSITIONS[nodeId] || { x: 0, y: 0, z: 0 };
    const svgX = ((pos.x + 4) / 10) * 360 + 20;
    const svgY = ((pos.z + 4) / 8) * 180 + 20;
    return { x: svgX, y: svgY };
  };

  return (
    <div style={{ background: '#0d1017', border: '1px solid var(--border)', padding: 8, borderRadius: 2 }}>
      <div style={{ fontSize: 10, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginBottom: 6 }}>
        CLICK NODES TO TARGET:
      </div>
      <svg width="100%" height="190" viewBox="0 0 400 220" style={{ display: 'block' }}>
        {topology.map((node) => {
          const { x, y } = mapCoord(node.nodeId);
          const isSelected = selectedNodes.includes(node.nodeId);

          return (
            <g
              key={node.nodeId}
              onClick={() => onToggleNode(node.nodeId)}
              style={{ cursor: 'pointer' }}
            >
              <rect
                x={x - 22}
                y={y - 14}
                width="44"
                height="28"
                rx="2"
                fill={isSelected ? 'rgba(217, 119, 6, 0.2)' : '#161923'}
                stroke={isSelected ? 'var(--accent)' : 'var(--border)'}
                strokeWidth={isSelected ? '2' : '1'}
              />
              <text
                x={x}
                y={y - 1}
                textAnchor="middle"
                fill={isSelected ? 'var(--accent)' : 'var(--text-1)'}
                fontSize="8.5"
                fontFamily="var(--font-mono)"
                fontWeight="600"
              >
                {node.nodeId}
              </text>
              <text
                x={x}
                y={y + 9}
                textAnchor="middle"
                fill="var(--text-3)"
                fontSize="6.5"
                fontFamily="var(--font-mono)"
              >
                {node.hostname.slice(0, 8)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
