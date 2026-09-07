// 12-node enterprise network topology
// same structure as network_simulator.py's create_default_network()

import type { NetworkNode, NodeStatus, NodeType } from './types';

// default network — called once per episode to get fresh state
export function createDefaultTopology(): NetworkNode[] {
  return structuredClone(DEFAULT_NODES);
}

const DEFAULT_NODES: NetworkNode[] = [
  {
    nodeId: 'fw-01',
    hostname: 'perimeter-fw',
    ip: '10.0.0.1',
    type: 'firewall',
    os: 'PfSense 2.7',
    status: 'healthy',
    openPorts: [443, 8443],
    services: ['firewall', 'vpn'],
    isCritical: true,
    vulnerabilities: [],
  },
  {
    nodeId: 'dc-01',
    hostname: 'ad-controller',
    ip: '10.0.1.1',
    type: 'domain_controller',
    os: 'Windows Server 2022',
    status: 'healthy',
    openPorts: [53, 88, 389, 636, 445],
    services: ['dns', 'kerberos', 'ldap', 'smb'],
    isCritical: true,
    vulnerabilities: [],
  },
  {
    nodeId: 'web-01',
    hostname: 'web-frontend',
    ip: '10.0.2.1',
    type: 'web_server',
    os: 'Ubuntu 22.04',
    status: 'healthy',
    openPorts: [80, 443, 22],
    services: ['nginx', 'nodejs', 'ssh'],
    isCritical: true,
    vulnerabilities: ['CVE-2024-1234-nginx'],
  },
  {
    nodeId: 'mail-01',
    hostname: 'mail-server',
    ip: '10.0.2.2',
    type: 'mail_server',
    os: 'Ubuntu 22.04',
    status: 'healthy',
    openPorts: [25, 587, 993, 22],
    services: ['postfix', 'dovecot', 'ssh'],
    isCritical: true,
    vulnerabilities: [],
  },
  {
    nodeId: 'app-01',
    hostname: 'app-backend',
    ip: '10.0.3.2',
    type: 'app_server',
    os: 'Ubuntu 22.04',
    status: 'healthy',
    openPorts: [8080, 8443, 22],
    services: ['java', 'tomcat', 'ssh'],
    isCritical: false,
    vulnerabilities: [],
  },
  {
    nodeId: 'db-01',
    hostname: 'prod-database',
    ip: '10.0.3.1',
    type: 'database',
    os: 'CentOS 9',
    status: 'healthy',
    openPorts: [5432, 22],
    services: ['postgresql', 'ssh'],
    isCritical: true,
    vulnerabilities: [],
  },
  {
    nodeId: 'backup-01',
    hostname: 'backup-svr',
    ip: '10.0.3.3',
    type: 'backup_server',
    os: 'Ubuntu 22.04',
    status: 'healthy',
    openPorts: [22, 873],
    services: ['ssh', 'rsync'],
    isCritical: true,
    vulnerabilities: [],
  },
  {
    nodeId: 'ws-01',
    hostname: 'analyst-pc-1',
    ip: '10.0.4.1',
    type: 'workstation',
    os: 'Windows 11',
    status: 'healthy',
    openPorts: [445, 3389],
    services: ['smb', 'rdp'],
    isCritical: false,
    vulnerabilities: [],
  },
  {
    nodeId: 'ws-02',
    hostname: 'dev-pc-1',
    ip: '10.0.4.2',
    type: 'workstation',
    os: 'Windows 11',
    status: 'healthy',
    openPorts: [445, 3389],
    services: ['smb', 'rdp'],
    isCritical: false,
    vulnerabilities: [],
  },
  {
    nodeId: 'ws-03',
    hostname: 'hr-pc-1',
    ip: '10.0.4.3',
    type: 'workstation',
    os: 'Windows 11',
    status: 'healthy',
    openPorts: [445, 3389],
    services: ['smb', 'rdp'],
    isCritical: false,
    vulnerabilities: [],
  },
  {
    nodeId: 'ws-04',
    hostname: 'exec-pc-1',
    ip: '10.0.4.4',
    type: 'workstation',
    os: 'macOS 14',
    status: 'healthy',
    openPorts: [22, 5900],
    services: ['ssh', 'vnc'],
    isCritical: false,
    vulnerabilities: [],
  },
  {
    nodeId: 'honey-01',
    hostname: 'honeypot-svr',
    ip: '10.0.5.1',
    type: 'honeypot',
    os: 'Debian 12',
    status: 'healthy',
    openPorts: [],
    services: [],
    isCritical: false,
    vulnerabilities: [],
  },
];

// which nodes can talk to each other
export const NETWORK_LINKS: ReadonlyArray<[string, string]> = [
  ['fw-01', 'web-01'],
  ['fw-01', 'dc-01'],
  ['fw-01', 'mail-01'],
  ['fw-01', 'db-01'],
  ['dc-01', 'mail-01'],
  ['dc-01', 'app-01'],
  ['app-01', 'db-01'],
  ['db-01', 'backup-01'],
  ['dc-01', 'ws-01'],
  ['dc-01', 'ws-02'],
  ['dc-01', 'ws-03'],
  ['dc-01', 'ws-04'],
  ['app-01', 'ws-03'],
];

// positions for rendering — used by both 2D and 3D views
// 3d uses x/z, 2d uses x/y
export const NODE_POSITIONS: Record<string, { x: number; y: number; z: number }> = {
  'fw-01':     { x: 0,   y: 0, z: -3 },
  'web-01':    { x: 2.5, y: 0, z: -3 },
  'dc-01':     { x: -3,  y: 0, z: 0 },
  'mail-01':   { x: -1,  y: 0, z: 0 },
  'app-01':    { x: 1,   y: 0, z: 0 },
  'db-01':     { x: 3,   y: 0, z: 0 },
  'backup-01': { x: 5,   y: 0, z: 0 },
  'ws-01':     { x: -3,  y: 0, z: 3 },
  'ws-02':     { x: -1,  y: 0, z: 3 },
  'ws-03':     { x: 1,   y: 0, z: 3 },
  'ws-04':     { x: 3,   y: 0, z: 3 },
  'honey-01':  { x: 5,   y: 0, z: 3 },
};

// node icons for 2D fallback
export const NODE_ICONS: Record<NodeType, string> = {
  firewall: '🔥',
  domain_controller: '🏛️',
  web_server: '🌐',
  mail_server: '📧',
  app_server: '⚙️',
  database: '🗄️',
  workstation: '💻',
  honeypot: '🍯',
  backup_server: '💾',
};

// status to color mapping
export const STATUS_COLORS: Record<NodeStatus, string> = {
  healthy: '#16a34a',
  compromised: '#dc2626',
  isolated: '#2563eb',
  offline: '#6b7280',
  encrypted: '#7c3aed',
  patched: '#0891b2',
};

// network state during an episode
export interface NetworkState {
  nodes: NetworkNode[];
  blockedIps: Set<string>;
  honeypotActive: boolean;
  isolatedNodes: Set<string>;
}

// create fresh network state for a new episode
export function createNetworkState(compromisedNodeIds: string[] = []): NetworkState {
  const nodes = createDefaultTopology();

  // mark initially compromised nodes
  for (const node of nodes) {
    if (compromisedNodeIds.includes(node.nodeId)) {
      node.status = 'compromised';
    }
  }

  return {
    nodes,
    blockedIps: new Set(),
    honeypotActive: false,
    isolatedNodes: new Set(),
  };
}

// get a node by id
export function getNode(state: NetworkState, nodeId: string): NetworkNode | undefined {
  return state.nodes.find(n => n.nodeId === nodeId);
}

// calculate overall network health (0-100)
export function getNetworkHealth(state: NetworkState): number {
  let healthyCount = 0;
  let total = 0;

  for (const node of state.nodes) {
    // skip honeypot
    if (node.type === 'honeypot') continue;
    total++;

    const weight = node.isCritical ? 2 : 1;
    if (node.status === 'healthy' || node.status === 'patched') {
      healthyCount += weight;
    } else if (node.status === 'isolated') {
      healthyCount += weight * 0.5; // isolated is better than compromised
    }
  }

  const maxWeight = state.nodes
    .filter(n => n.type !== 'honeypot')
    .reduce((sum, n) => sum + (n.isCritical ? 2 : 1), 0);

  return Math.round((healthyCount / maxWeight) * 100);
}

// get current threat level based on node statuses
export function getThreatLevel(state: NetworkState): 'green' | 'yellow' | 'orange' | 'red' | 'critical' {
  const compromised = state.nodes.filter(
    n => n.status === 'compromised' || n.status === 'encrypted'
  );
  const criticalCompromised = compromised.filter(n => n.isCritical);

  if (criticalCompromised.length >= 3) return 'critical';
  if (criticalCompromised.length >= 1) return 'red';
  if (compromised.length >= 3) return 'orange';
  if (compromised.length >= 1) return 'yellow';
  return 'green';
}

// apply an agent action to the network state
export function applyAction(
  state: NetworkState,
  tool: string,
  args: Record<string, string>,
): { success: boolean; cost: number; description: string } {
  switch (tool) {
    case 'observe_network':
      return { success: true, cost: 0, description: 'Network observation complete.' };

    case 'investigate_alert':
      return { success: true, cost: 2, description: `Investigating alert ${args.alert_id}.` };

    case 'run_forensics': {
      const node = getNode(state, args.node_id);
      if (!node) return { success: false, cost: 0, description: `Node ${args.node_id} not found.` };
      return { success: true, cost: 5, description: `Deep forensic scan on ${args.node_id}.` };
    }

    case 'block_ip': {
      const ip = args.ip_address;
      if (state.blockedIps.has(ip)) {
        return { success: false, cost: 0, description: `IP ${ip} already blocked.` };
      }
      state.blockedIps.add(ip);
      return { success: true, cost: 1, description: `Blocked IP ${ip} at firewall.` };
    }

    case 'isolate_host': {
      const node = getNode(state, args.node_id);
      if (!node) return { success: false, cost: 0, description: `Node ${args.node_id} not found.` };
      if (state.isolatedNodes.has(args.node_id)) {
        return { success: false, cost: 0, description: `${args.node_id} already isolated.` };
      }
      node.status = 'isolated';
      state.isolatedNodes.add(args.node_id);
      return { success: true, cost: 3, description: `Isolated host ${args.node_id}.` };
    }

    case 'dismiss_alert':
      return { success: true, cost: 1, description: `Dismissed alert ${args.alert_id}.` };

    case 'restore_backup': {
      const node = getNode(state, args.node_id);
      if (!node) return { success: false, cost: 0, description: `Node ${args.node_id} not found.` };
      node.status = 'healthy';
      node.vulnerabilities = [];
      state.isolatedNodes.delete(args.node_id);
      return { success: true, cost: 8, description: `Restored ${args.node_id} from backup.` };
    }

    case 'deploy_patch': {
      const node = getNode(state, args.node_id);
      if (!node) return { success: false, cost: 0, description: `Node ${args.node_id} not found.` };
      node.status = 'patched';
      node.vulnerabilities = [];
      return { success: true, cost: 3, description: `Deployed patch to ${args.node_id}.` };
    }

    case 'deploy_honeypot':
      state.honeypotActive = true;
      return { success: true, cost: 4, description: 'Honeypot deployed.' };

    case 'escalate_incident':
      return { success: true, cost: 5, description: 'Incident escalated to senior team.' };

    default:
      return { success: false, cost: 0, description: `Unknown action: ${tool}` };
  }
}
