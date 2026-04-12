---
title: CyberRange
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

<h1 align="center">CyberRange</h1>
<p align="center"><strong>Reinforcement Learning Environment for Autonomous Security Operations Center (SOC) Agents</strong></p>

<p align="center">
  <a href="https://github.com/meta-pytorch/OpenEnv">Built with OpenEnv v0.2.2</a> |
  <a href="https://huggingface.co/spaces/keshav-005/cyber_range">HuggingFace Spaces Deployment</a> |
  <a href="https://github.com/huggingface/trl">Training via HF TRL</a>
</p>

<hr>

## Project Overview

CyberRange is an advanced, self-improving simulation environment designed to train and benchmark autonomous security agents. By acting as a Security Operations Center (SOC) analyst, an agent interfaces with a simulated enterprise network spanning 12 distinct topological nodes. The agent processes inbound Security Information and Event Management (SIEM) alerts without a priori knowledge of which alerts are benign or malicious.

Utilizing adversarial attack generation, dynamic curriculum scaling, and Group Relative Policy Optimization (GRPO), CyberRange forces the reinforcement learning agent to learn essential SOC fundamentals: precise telemetry investigation, rapid containment, false-positive mitigation, and infrastructure restoration.

## Core Lifecycle

### 1. The Cold Start
Initially, the agent lacks structural awareness. Given an alert indicating a critical brute-force attack on a web server, the untrained agent may impulsively isolate benign infrastructure or indiscriminately block traffic, resulting in severe downtime penalties and a fractional baseline score.

### 2. Behavioral Adaptation
Through iterative episodes and repeat-command penalization, the agent inherently learns to sequence its workflows. It learns the critical dependency of invoking investigation tools to surface forensic evidence before instituting perimeter firewall blocks. Performance scales as the agent strategically dismisses false positives before committing to containment.

### 3. Adversarial Engineering
As the agent achieves baseline mastery, the environment introduces the **Adversarial Attack Designer**. This module autonomously architects complex, multi-stage APT (Advanced Persistent Threat) chains explicitly targeting the agent's operational blind spots. Simultaneously, the **Curriculum Manager** scales difficulty progressively, guaranteeing that the training distribution adapts in real-time to the agent's competency.

### 4. Multi-Persona Evaluation
Every completed episode is evaluated by a deterministic engine in tandem with **CyberJudge**—a triad of LLM expert personas observing the episode logs:
* **Junior Analyst**: Evaluates investigative completeness and procedural integrity.
* **Senior Analyst**: Assesses triage prioritization and false-positive handling efficiency.
* **Incident Commander**: Measures strategic incident containment and operational blast radius.

The final evaluation (70% deterministic, 30% consensus heuristic) yields the high-variance, dense reward gradient required for stable GRPO convergence.

<hr>

## Logical Architecture

```text
+-------------------------------------------------------------------------+
|                        SELF-IMPROVING LOOP                              |
|                                                                         |
|  [Adversarial]      [Enterprise  ]      [ Autonomous ]      [CyberJudge ]
|  [ Designer  ] ---> [  Network   ] ---> [   Agent    ] ---> [ Evaluation]
|  [(LLM-based)]      [(12 nodes)  ]      [(LLM/Rules) ]      [(LLM Panel)]
|       ^                   |                   |                  |      |
|       |                   |                   v                  |      |
|       |             [Curriculum  ]      [   Reward   ] <---------+      |
|       +------------ [ Manager    ]      [ Generation ]                  |
|     weak spots &    [ (Tracking) ]            |                         |
|     difficulty            |                   v                         |
|                           +--------> GRPO gradient update               |
+-------------------------------------------------------------------------+
```

### Innovative Mechanisms
* **Self-Generating Scenarios**: Attack topologies are dynamically synthesized to target operational vulnerabilities. No manual scenario authoring is required.
* **Persistent Skill Library**: Agents archive high-yield investigation procedures into a SQLite-backed Full-Text Search protocol to emulate compounding institutional knowledge.
* **Multi-Dimensional Signal**: High-variance scoring combines repeat penalties, strict phase-order adherence, efficiency scaling, and timeout mechanics.
* **MITRE ATT&CK Framework Synchronization**: Every vector is logged with corresponding global defense framework identifiers.

<hr>

## Threat Topologies

| Scenario Matrix | Difficulty Designation | Primary Threat Vectors | MITRE ATT&CK Taxonomy |
|---|---|---|---|
| `script_kiddie` | Entry | Reconnaissance, Baseline Exploitation | T1078, T1110 |
| `phishing_campaign` | Intermediate | Phishing, Credential Exfiltration | T1566.001, T1003 |
| `apt_lateral_movement` | Advanced | Pivot Exploitation, Privilege Escalation | T1190, T1068, T1021, T1041 |
| `ransomware_outbreak` | Advanced | Ransomware Detonation, Drive-by Spread | T1566, T1486, T1021 |
| `supply_chain_compromise` | Advanced | Trojan Updates, Command & Control (C2) | T1195.002, T1059, T1105, T1041 |
| `insider_threat_apt` | Nightmare | Simultaneous Exfiltration & APT Operations | T1078, T1567, T1190, T1003 |

*(Supplementary scenarios are infinitely instantiated via the integrated Adversarial Attack Designer.)*

<hr>

## Development and Training Signal

The reward synthesis engine provides strict reinforcement bounds:
* **Per-Step Return**: Positive alignment for threat containment; drastic negative alignment for collateral system isolation.
* **Structural Penalization**: Negative reinforcement strictly applied for redundant action loops to heavily mitigate reward hacking.
* **Workflow Integrity Bonus**: Strong reinforcement scaling for abiding by standard protocol: Triage -> Consolidate -> Contain.

### Baseline Evaluation Data
Initial metrics tracking an optimized heuristic algorithm via `eval.py`:

| Test Definition | Total Score | Efficiency Ratio | Episode Yield |
|---|---|---|---|
| `script_kiddie` | 0.800 | 3/15 | 20.5 |
| `phishing_campaign` | 0.450 | 11/25 | 86.7 |
| `apt_lateral_movement` | 0.420 | 19/35 | 125.0 |
| `ransomware_outbreak` | 0.650 | 10/20 | 89.5 |
| `supply_chain_compromise` | 0.420 | 16/30 | 121.8 |
| `insider_threat_apt` | 0.400 | 19/45 | 125.0 |

<hr>

## System Toolchain Reference

Agents interact strictly via standardized Model Context Protocol integrations.

| Operation Context | Capability Outline | Application Cost |
|---|---|---|
| `observe_network` | Retrieves the absolute state of the network status, alert queue, and telemetry scores. | 0 |
| `investigate_alert` | Deep-link forensic review to isolate specific threat variables and origin IP data. | 2 |
| `run_forensics` | Advanced host memory and localized storage vulnerability validation. | 5 |
| `block_ip` | Egress boundary blockage for inbound connections at the perimeter gateway. | 1 |
| `isolate_host` | Complete network demarcation for the specified node. | 3 |
| `dismiss_alert` | Standardize validation of procedural noise or administrative benign triggers. | 1 |
| `restore_backup` | Heavy system refresh overriding existing storage from remote secondary. | 8 |
| `deploy_patch` | Standardize immediate vulnerability fix deployments. | 3 |
| `deploy_honeypot` | Setup decoy configurations to monitor extensive unauthorized internal traversal. | 4 |
| `escalate_incident` | Flag scenario for manual secondary responder interaction. | 5 |
| `save_playbook` | Consolidate successful actions into a queryable skill index loop. | 0 |
| `search_playbooks` | Compare currently visible vectors against archived strategy data arrays. | 0 |

<hr>

## Local Deployment Operations

```bash
# Evaluate internal heuristic baseline across all loaded scenarios
python eval.py --save

# Generate visualization parameters for simulated performance scaling
python plot_rewards.py --simulate --episodes 50

# Execute internal testing bounds
python -m pytest tests/ -q
```

## Cloud Environment Formatting

Native support is provided for generalized deployment on containerized environments such as HuggingFace Spaces.

```dockerfile
# OpenEnv foundational reference image required
FROM ghcr.io/meta-pytorch/openenv-base:latest
COPY . /app
CMD ["uvicorn", "cyber_range.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Operating Configuration Variables

| Value Identifier | Required | Scope Overview |
|---|---|---|
| `MODEL_NAME` | True (LLM enabled) | Central reasoning model definition structure |
| `API_BASE_URL` | True (LLM enabled) | Inference hosting path |
| `HF_TOKEN` | False | Authentication gateway key for Huggingface resources |
| `OPENAI_API_KEY` | False | Alternate LLM structural authorization |

*(If external APIs are invalid, the foundational logic engine defaults back gracefully to localized deterministic frameworks allowing uninterrupted workflow.)*
