// grading + cyberjudge persona verdicts
// deterministic scoring with template-based feedback

import type { GraderResult, AttackPhase, Alert } from './types';
import type { NetworkState } from './network';
import { calculateFinalScore } from './rewards';

// persona verdict templates indexed by score range
const JUNIOR_VERDICTS: Record<string, string[]> = {
  high: [
    'All alerts investigated before containment. Proper SOC procedure followed.',
    'Thorough investigation phase. No alerts were skipped or ignored.',
    'Good evidence collection. Every alert got reviewed before action.',
  ],
  mid: [
    'Most alerts investigated but some were acted on prematurely.',
    'Decent investigation coverage but triage order could improve.',
    'Some alerts were skipped. Need to verify all before containment.',
  ],
  low: [
    'Multiple alerts were ignored. Incomplete investigation.',
    'Jumped to containment without proper evidence review.',
    'Poor investigation discipline. Several critical alerts unreviewed.',
  ],
};

const SENIOR_VERDICTS: Record<string, string[]> = {
  high: [
    'Correct triage priority — critical alerts first. Efficient FP dismissal.',
    'Excellent severity-based prioritization. Clean false positive handling.',
    'Strong triage instincts. Budget well-allocated across actions.',
  ],
  mid: [
    'Triage order was acceptable but not optimal. Some budget waste.',
    'Decent prioritization. Could be more aggressive on FP dismissal.',
    'Mixed efficiency. Good threat detection but slow FP resolution.',
  ],
  low: [
    'Poor triage order. Low-severity alerts investigated before critical.',
    'False positives consumed too much time. Budget mismanagement.',
    'Needs significant improvement in alert prioritization.',
  ],
};

const COMMANDER_VERDICTS: Record<string, string[]> = {
  high: [
    'Minimal blast radius. Business continuity maintained. Swift containment.',
    'Excellent strategic response. Critical systems protected. Data secured.',
    'Clean incident response. No unnecessary collateral damage.',
  ],
  mid: [
    'Acceptable containment but response time was concerning.',
    'Blast radius manageable. Some business disruption from over-isolation.',
    'Threats contained but data protection could have been faster.',
  ],
  low: [
    'Excessive collateral damage. Healthy systems isolated unnecessarily.',
    'Slow response allowed significant data exfiltration.',
    'Unacceptable business impact. Need faster, more targeted containment.',
  ],
};

function pickVerdict(templates: Record<string, string[]>, normalizedScore: number): string {
  const tier = normalizedScore >= 0.7 ? 'high' : normalizedScore >= 0.4 ? 'mid' : 'low';
  const options = templates[tier];
  // deterministic pick based on score to avoid randomness in replay
  const idx = Math.floor(normalizedScore * 100) % options.length;
  return options[idx];
}

// grade an episode and generate judge verdicts
export function gradeEpisode(
  phases: AttackPhase[],
  alerts: Alert[],
  networkState: NetworkState,
  stepsUsed: number,
  maxSteps: number,
  actionHistory: string[],
  adversaryBehavior: string,
): GraderResult {
  const scores = calculateFinalScore(
    phases, alerts, networkState, stepsUsed, maxSteps, actionHistory,
  );

  // normalize component scores for verdict selection
  const threatNorm = scores.threatResponse / 0.35;
  const fpNorm = scores.falsePositiveHandling / 0.20;
  const effNorm = scores.efficiency / 0.10;

  // junior cares about investigation completeness
  const juniorScore = Math.round((threatNorm * 0.4 + fpNorm * 0.4 + effNorm * 0.2) * 100) / 100;

  // senior cares about triage efficiency
  const seniorScore = Math.round((fpNorm * 0.5 + effNorm * 0.3 + threatNorm * 0.2) * 100) / 100;

  // commander cares about blast radius and data protection
  const collateralNorm = scores.collateralDamage / 0.15;
  const dataNorm = scores.dataProtection / 0.20;
  const commanderScore = Math.round((collateralNorm * 0.4 + dataNorm * 0.4 + threatNorm * 0.2) * 100) / 100;

  return {
    finalScore: scores.finalScore,
    threatResponse: scores.threatResponse,
    falsePositiveHandling: scores.falsePositiveHandling,
    dataProtection: scores.dataProtection,
    collateralDamage: scores.collateralDamage,
    efficiency: scores.efficiency,
    details: {
      threatsNeutralized: scores.details.threatsNeutralized as number,
      totalThreats: scores.details.totalThreats as number,
      fpDismissed: scores.details.fpDismissed as number,
      totalFps: scores.details.totalFps as number,
      stepsUsed: scores.details.stepsUsed as number,
      maxSteps: scores.details.maxSteps as number,
      dataExfiltratedMb: scores.details.dataExfiltratedMb as number,
      healthyHostsIsolated: scores.details.healthyHostsIsolated as number,
      adversaryBehavior,
    },
    judgeVerdicts: {
      junior: { score: juniorScore, verdict: pickVerdict(JUNIOR_VERDICTS, juniorScore) },
      senior: { score: seniorScore, verdict: pickVerdict(SENIOR_VERDICTS, seniorScore) },
      commander: { score: commanderScore, verdict: pickVerdict(COMMANDER_VERDICTS, commanderScore) },
    },
  };
}
