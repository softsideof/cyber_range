"""
CyberJudge — Multi-Persona LLM Evaluation Panel.

Three expert SOC personas grade the agent's episode performance,
producing a richer feedback signal than the deterministic grader.

Inspired by: Kube SRE Gym (Gold, OpenEnv SF Hackathon 2026).

Usage:
    judge = CyberJudge()
    result = judge.evaluate(episode_log)
    # result["llm_judge_score"]        → float 0.0–1.0
    # result["persona_scores"]         → per-persona breakdown
    # result["combined_score"]         → 0.7 * deterministic + 0.3 * llm
    # result["feedback"]               → actionable feedback string
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Persona definitions — each sees the same episode log but
# evaluates through a different professional lens
# ─────────────────────────────────────────────────────────────

PERSONAS: dict[str, str] = {
    "junior_analyst": """You are a Junior SOC Analyst with 1 year of experience.
You focus on: Did the agent investigate ALL alerts before acting?
Did it follow standard triage procedures? Were any alerts missed?
You are strict about thoroughness and procedure adherence.""",

    "senior_analyst": """You are a Senior SOC Analyst with 8 years of experience.
You focus on: Was the alert TRIAGE ORDER correct (high severity first)?
Were false positives handled efficiently without wasting investigation budget?
Did the agent correctly distinguish real threats from noise?
You value both accuracy and efficiency.""",

    "incident_commander": """You are a CISO-level Incident Commander.
You focus on: Was business impact minimized? Was the blast radius contained quickly?
Did the agent avoid unnecessary disruption to healthy systems?
Was data exfiltration prevented? Was the response proportionate to the threat?
You care about strategic outcomes, not just technical steps.""",
}

JUDGE_PROMPT_TEMPLATE = """You are evaluating a SOC analyst AI agent's performance on a cybersecurity incident response episode.

YOUR PERSONA:
{persona}

SCENARIO:
- Name: {scenario_name}
- Difficulty: {difficulty}
- Adversary: {adversary_behavior}
- Threats: {threat_count} real threats, {fp_count} false positives

EPISODE OUTCOME:
- Steps used: {steps_used}/{max_steps}
- Threats neutralized: {threats_neutralized}/{total_threats}
- False positives correctly dismissed: {fps_dismissed}/{fps_total}
- Data exfiltrated: {data_exfil_mb} MB
- Healthy hosts isolated (collateral damage): {healthy_isolated}
- Deterministic score: {det_score:.3f}

AGENT ACTION SEQUENCE:
{action_log}

EVALUATION INSTRUCTIONS:
Score the agent from 0.0 to 1.0 based on YOUR PERSONA's priorities.
Be critical but fair. Consider:
1. What did the agent do well?
2. What critical mistakes did it make?
3. Would a real SOC analyst be satisfied with this response?

Respond with ONLY valid JSON:
{{
  "score": <float 0.0-1.0>,
  "strengths": ["<max 2 bullet points>"],
  "weaknesses": ["<max 2 bullet points>"],
  "verdict": "<one sentence summary>"
}}"""


class CyberJudge:
    """
    Multi-Persona LLM Evaluation Panel for SOC episode grading.

    Falls back gracefully to deterministic score if no LLM API is configured.
    Set MODEL_NAME and API_BASE_URL environment variables to enable.
    """

    def __init__(self) -> None:
        self._api_base = os.getenv("API_BASE_URL", "").rstrip("/")
        self._model = os.getenv("MODEL_NAME", "")
        self._token = os.getenv("HF_TOKEN", "") or os.getenv("OPENAI_API_KEY", "")
        self._enabled = bool(self._api_base and self._model and self._token)

    def evaluate(
        self,
        episode_log: list[dict],
        scenario_meta: dict,
        deterministic_score: float,
    ) -> dict:
        """
        Run all 3 personas over the episode and return combined score.

        Args:
            episode_log:         List of {step, action, args, reward} dicts
            scenario_meta:       Scenario metadata from grader_result["details"]
            deterministic_score: The rule-based grader's final_score

        Returns:
            {
                "llm_judge_score": float,
                "combined_score":  float,   # 0.7 * det + 0.3 * llm
                "persona_scores":  dict,
                "feedback":        str,
                "judge_enabled":   bool,
            }
        """
        if not self._enabled or not episode_log:
            return self._fallback(deterministic_score)

        persona_results: dict[str, dict] = {}

        for persona_name, persona_desc in PERSONAS.items():
            result = self._evaluate_persona(
                persona_name, persona_desc, episode_log, scenario_meta, deterministic_score
            )
            persona_results[persona_name] = result

        # Average the 3 persona scores
        valid_scores = [
            r["score"] for r in persona_results.values()
            if isinstance(r.get("score"), (int, float))
        ]
        llm_score = sum(valid_scores) / len(valid_scores) if valid_scores else deterministic_score

        # Weighted combination: 70% deterministic + 30% LLM judge
        combined = round(0.70 * deterministic_score + 0.30 * llm_score, 4)

        # Aggregate feedback from all personas
        verdicts = [
            f"[{name.replace('_', ' ').title()}]: {r.get('verdict', '')}"
            for name, r in persona_results.items()
            if r.get("verdict")
        ]
        feedback = " | ".join(verdicts)

        return {
            "llm_judge_score": round(llm_score, 4),
            "combined_score": combined,
            "persona_scores": {
                name: r.get("score", 0.0)
                for name, r in persona_results.items()
            },
            "persona_feedback": {
                name: {
                    "strengths": r.get("strengths", []),
                    "weaknesses": r.get("weaknesses", []),
                    "verdict": r.get("verdict", ""),
                }
                for name, r in persona_results.items()
            },
            "feedback": feedback,
            "judge_enabled": True,
        }

    def _evaluate_persona(
        self,
        persona_name: str,
        persona_desc: str,
        episode_log: list[dict],
        meta: dict,
        det_score: float,
    ) -> dict:
        """Call LLM for a single persona evaluation."""
        action_log = self._format_action_log(episode_log)

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            persona=persona_desc,
            scenario_name=meta.get("scenario_name", "Unknown"),
            difficulty=meta.get("difficulty", "unknown"),
            adversary_behavior=meta.get("adversary_behavior", "unknown"),
            threat_count=meta.get("threat_count", "?"),
            fp_count=meta.get("false_positive_count", "?"),
            steps_used=meta.get("steps_used", "?"),
            max_steps=meta.get("max_steps", "?"),
            threats_neutralized=meta.get("threats_neutralized", "?"),
            total_threats=meta.get("threats_total", "?"),
            fps_dismissed=meta.get("fps_dismissed", "?"),
            fps_total=meta.get("fps_total", "?"),
            data_exfil_mb=meta.get("data_exfiltrated_mb", 0),
            healthy_isolated=meta.get("healthy_hosts_isolated", 0),
            det_score=det_score,
            action_log=action_log,
        )

        try:
            raw = self._call_llm(prompt)
            parsed = self._parse_json_response(raw)
            # Clamp score to valid range
            parsed["score"] = max(0.0, min(1.0, float(parsed.get("score", det_score))))
            return parsed
        except Exception as e:
            return {"score": det_score, "verdict": f"[Judge error: {e}]"}

    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM API (OpenAI-compatible)."""
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package not installed")

        client = OpenAI(
            base_url=self._api_base + "/v1",
            api_key=self._token,
        )

        # Rate limiting — avoid hammering the API 3x per episode
        time.sleep(0.5)

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert cybersecurity evaluator. Always respond with valid JSON only.",
                },
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown fences."""
        text = text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise ValueError(f"No JSON found in response: {text[:100]}")

    def _format_action_log(self, episode_log: list[dict]) -> str:
        """Format episode log into readable string for the LLM."""
        lines = []
        for entry in episode_log:
            step = entry.get("step", "?")
            action = entry.get("action", "?")
            args = entry.get("args", {})
            reward = entry.get("reward", 0.0)
            reward_str = f"+{reward:.2f}" if reward >= 0 else f"{reward:.2f}"
            args_str = json.dumps(args) if args else "{}"
            lines.append(f"  Step {step:>2}: {action}({args_str}) → reward {reward_str}")
        return "\n".join(lines) if lines else "  (no actions recorded)"

    def _fallback(self, det_score: float) -> dict:
        """Return fallback result when LLM is not configured."""
        return {
            "llm_judge_score": det_score,
            "combined_score": det_score,
            "persona_scores": {},
            "persona_feedback": {},
            "feedback": "LLM judge not configured (set MODEL_NAME + API_BASE_URL + HF_TOKEN).",
            "judge_enabled": False,
        }


# ─────────────────────────────────────────────────────────────
# Episode log builder — collect actions during inference
# ─────────────────────────────────────────────────────────────

class EpisodeLogger:
    """Collects tool calls and rewards during an episode for judge input."""

    def __init__(self) -> None:
        self.log: list[dict] = []

    def record(self, step: int, action: str, args: dict, reward: float) -> None:
        self.log.append({"step": step, "action": action, "args": args, "reward": reward})

    def reset(self) -> None:
        self.log = []

    def to_judge_format(self) -> list[dict]:
        return list(self.log)
