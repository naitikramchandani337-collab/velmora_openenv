---
title: Velmora Incident Response OpenEnv
emoji: 🚨
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Velmora Incident Response OpenEnv

Velmora Incident Response OpenEnv is a real-world simulation environment for evaluating AI agents on incident management and operational troubleshooting tasks.

The environment models how engineers respond to issues such as outages, service degradation, integration failures, and security-related incidents. The goal is to evaluate whether an agent can make sensible operational decisions under limited resources.

## Why This Matters

Incident response is a real task performed by SREs, DevOps engineers, platform teams, and security responders.

This environment tests whether an AI agent can:

- investigate incidents effectively
- apply the right action sequence
- contain severe incidents early
- escalate when external support is needed
- use limited actions efficiently
- avoid repeated or wasteful actions

## Tasks

The environment contains three difficulty levels:

### Easy
Low-impact incidents such as UI errors and minor frontend failures.

### Medium
Service issues such as API latency, intermittent integration failures, and unstable subsystems that require multi-step reasoning.

### Hard
High-severity incidents such as breach suspicion, cascading service failures, and traffic anomalies requiring containment, escalation, and strategic recovery.

## Action Space

The agent may choose one of the following actions:

- `investigate`
- `fix`
- `monitor`
- `escalate`
- `contain`

## Observation Space

Each observation includes:

- `incident` — the incident description
- `hint` — contextual clue for diagnosis
- `logs` — simulated system logs
- `user_impact` — impact severity
- `system_status` — operational health state
- `steps_taken` — action history
- `resources` — remaining action budget
- `severity` — low / medium / high
- `current_stage` — inferred incident handling stage

## Reward Design

The reward is dense and shaped. It is not just a binary success signal.

It includes:

- progress toward incident resolution
- correctness of chosen actions
- sequencing quality
- resource efficiency
- severity-aware penalties
- penalties for repeated, unsafe, or wasteful actions

Reward scores are normalized in the range **0.0–1.0**.

## Grading

The environment includes deterministic task-level grading for:

- `easy`
- `medium`
- `hard`

Each task returns a normalized score in the range **0.0–1.0**.

## OpenEnv Interface

The environment supports:

- `reset()`
- `step(action)`
- `state()`

The server exposes corresponding API endpoints.

## API Endpoints

- `GET /` — service info
- `GET /health` — health check
- `POST /reset` — start a new episode
- `POST /step` — execute an action
- `GET /state` — inspect current environment state
- `GET /tasks` — list supported tasks and actions
- `GET /grader` — return current task score

## Inference Script

The required root-level inference file is:

- `inference.py`

It uses the following environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Example:

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your_token_here"
python inference.py