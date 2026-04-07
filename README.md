# Velmora Incident Response OpenEnv

## Overview

Velmora Incident Response OpenEnv is a real-world simulation environment designed to evaluate AI agents on incident management tasks. The environment models how engineers respond to system issues such as outages, performance degradation, and security threats.

The goal is to assess an agent’s ability to make correct decisions, prioritize actions, and efficiently resolve incidents under resource constraints.

---

## Real-World Motivation

Incident response is a critical task in software engineering, DevOps, and site reliability engineering (SRE). This environment captures realistic workflows including investigation, containment, escalation, and monitoring.

It provides a structured benchmark for training and evaluating autonomous agents in operational decision-making scenarios.

---

## Tasks

The environment includes three difficulty levels:

### Easy
S

---

## Action Space

Discrete actions available to the agent:

- investigate
- fix
- monitor
- escalate
- contain

---

## Observation Space

Each observation includes:

- incident: description of the problem
- hint: contextual clue for diagnosis
- logs: simulated system logs
- user_impact: severity of impact
- system_status: current system health
- steps_taken: history of actions
- resources: remaining action budget

---

## Reward Function

The reward function provides continuous feedback based on:

- progress toward resolution
- efficiency of resource usage
- correctness of actions
- severity-aware weighting
- penalties for incorrect, repeated, or poorly sequenced actions

Scores are normalized between 0.0 and 1.0.

---

## Agent Behavior

The baseline agent uses a structured prompt to simulate real-world incident response reasoning.

The prompt enforces:

- correct action sequencing (investigate → fix → monitor)
- handling of high-severity incidents (contain → investigate → escalate)
- avoidance of repeated or wasteful actions

This ensures that the agent behaves like a realistic system operator rather than a random decision-maker.

---

## Baseline

Run the baseline agent:

```bash
python -m baseline.run_baseline