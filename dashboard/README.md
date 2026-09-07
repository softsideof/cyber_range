# CyberRange — Interactive 3D Autonomous SOC Defense Platform

> **Live Interactive Threat Simulation & Autonomous Policy Evaluation**  
> Built as an interactive showcase for AI Developer / Research roles. Deploys to Vercel with zero server dependencies and runs seamlessly on mobile devices.

---

## 🎯 Project Overview

CyberRange simulates an enterprise network under adversarial attack (APTs, ransomware worms, supply chain trojans). An autonomous reinforcement learning policy (trained via GRPO conforming to the OpenEnv standard) observes real-time SIEM alerts, investigates forensic evidence, isolates compromised hosts, blocks adversary C2 infrastructure, and defends the network against multi-stage attacks.

### Key Engineering Features

- **Client-Side Simulation Engine**: High-performance TypeScript port of the Python RL training environment. Runs in Web Workers off the main thread with an automatic direct client fallback.
- **Interactive 3D Network Graph**: Three.js & React Three Fiber spatial visualization with specialized 3D geometries per node type (firewalls, domain controllers, database servers, workstations).
- **Adversary Progression & Behavioral Modes**: Simulates static, evasive (C2 IP rotation), persistent (backdoor re-infection), and adaptive (decoy alert injection) adversary doctrines.
- **Weaponized Virus Constructor**: Interviewers can launch pre-built virus campaigns (**WannaCry Lite**, **SolarWinds Jr.**, **Ghost Operator**) or compose custom multi-stage attacks using an interactive 2D node target selector.
- **CyberJudge Evaluation**: Deterministic 5-component weighted scoring (Threat Response 35%, FP Handling 20%, Data Protection 20%, Collateral Damage 15%, Efficiency 10%) synthesized into qualitative persona feedback from Junior Analyst, Senior Lead, and Incident Commander perspectives.
- **Anti-AI Operational UI**: Strict CrowdStrike/Splunk-inspired aesthetic with `#0f1117` matte dark surfaces, JetBrains Mono typography, dense SIEM feeds, and asymmetric layout.

---

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm

### Local Development
```bash
# Install dependencies
npm install --legacy-peer-deps

# Run simulation engine unit tests
npm test

# Start local Next.js development server
npm run dev
# Open http://localhost:3000 in your browser
```

### Production Build & Deploy
```bash
# Production bundle build
npm run build

# Deploy to Vercel
npx vercel --prod
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| <kbd>Space</kbd> | Pause / Resume simulation |
| <kbd>B</kbd> | Open Adversarial Virus Builder |
| <kbd>A</kbd> | Toggle System Architecture Specification |
| <kbd>1</kbd>–<kbd>6</kbd> | Jump directly to built-in attack scenarios |
| <kbd>Esc</kbd> | Close active drawers and overlays |
