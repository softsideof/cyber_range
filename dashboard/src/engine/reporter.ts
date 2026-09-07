// automated incident post-mortem / after-action report (AAR) generator
// formats episode defense forensics into a structured markdown document

import type { GraderResult, ScenarioConfig, AgentLogEntry, NetworkNode } from './types';

export function generateIncidentReport(
  graderResult: GraderResult,
  scenario: ScenarioConfig | null,
  agentLogs: AgentLogEntry[] = [],
  nodes: NetworkNode[] = [],
): string {
  const scorePct = Math.round(graderResult.finalScore * 100);
  const now = new Date().toISOString();

  const lines: string[] = [
    `# Security Incident Post-Mortem & After-Action Report (AAR)`,
    ``,
    `**Incident Scenario:** ${scenario?.name || 'Custom Cyber Attack'}  `,
    `**Target Architecture:** 12-Node Enterprise Perimeter & LAN  `,
    `**Generated Timestamp:** ${now}  `,
    `**Overall IR Score:** ${scorePct}% (Weighted Multi-Objective)  `,
    `**Adversary Behavior:** ${graderResult.details.adversaryBehavior.toUpperCase()}  `,
    ``,
    `---`,
    ``,
    `## 1. Executive Summary & Readiness Assessment`,
    ``,
    `The autonomous SOC defense agent engaged an active adversarial intrusion campaign across ${graderResult.details.stepsUsed} operational intervals (max budget: ${graderResult.details.maxSteps}).`,
    ``,
    `* **Threat Containment Rate:** ${graderResult.details.threatsNeutralized} / ${graderResult.details.totalThreats} active vectors neutralized.`,
    `* **False Positive Triage:** ${graderResult.details.fpDismissed} / ${graderResult.details.totalFps} benign alerts validated and dismissed without business disruption.`,
    `* **Data Exfiltration:** ${graderResult.details.dataExfiltratedMb} MB exfiltrated before containment.`,
    `* **Operational Blast Radius:** ${graderResult.details.healthyHostsIsolated} non-infected systems isolated.`,
    ``,
    `---`,
    ``,
    `## 2. CyberJudge Triad Evaluation`,
    ``,
    `### Junior SOC Analyst (Procedural Integrity: ${Math.round(graderResult.judgeVerdicts.junior.score * 100)}%)`,
    `> "${graderResult.judgeVerdicts.junior.verdict}"`,
    ``,
    `### Senior Incident Lead (Triage & Cost Efficiency: ${Math.round(graderResult.judgeVerdicts.senior.score * 100)}%)`,
    `> "${graderResult.judgeVerdicts.senior.verdict}"`,
    ``,
    `### Incident Commander (Blast Radius & Business Continuity: ${Math.round(graderResult.judgeVerdicts.commander.score * 100)}%)`,
    `> "${graderResult.judgeVerdicts.commander.verdict}"`,
    ``,
    `---`,
    ``,
    `## 3. Incident Timeline & Tactical Defense Audit`,
    ``,
    `| Step | Action Taken | Target / Target IP | Result | Cost |`,
    `| :--- | :--- | :--- | :--- | :--- |`,
  ];

  if (agentLogs.length === 0) {
    lines.push(`| - | No incident timeline recorded | - | - | - |`);
  } else {
    agentLogs.forEach((log) => {
      const target = log.action.args.host || log.action.args.ip || log.action.args.alert_id || 'System';
      const cost = `${log.result.cost} pts`;
      const desc = log.result.description.replace(/\|/g, '-');
      lines.push(`| ${log.step} | \`${log.action.tool}\` | ${target} | ${desc} | ${cost} |`);
    });
  }

  lines.push(
    ``,
    `---`,
    ``,
    `## 4. Host Infrastructure Status Post-Containment`,
    ``,
    `| Node ID | Hostname | Role | Final Operational Status |`,
    `| :--- | :--- | :--- | :--- |`,
  );

  nodes.forEach((n) => {
    lines.push(`| \`${n.nodeId}\` | ${n.hostname} | ${n.type} | **${n.status.toUpperCase()}** |`);
  });

  lines.push(
    ``,
    `---`,
    ``,
    `*Report generated automatically by CyberRange OpenEnv Autonomous SOC Simulation Engine.*`,
  );

  return lines.join('\n');
}

export function downloadIncidentReport(reportMarkdown: string, filename = 'SOC_Incident_Report.md'): void {
  if (typeof window === 'undefined') return;
  const blob = new Blob([reportMarkdown], { type: 'text/markdown;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
