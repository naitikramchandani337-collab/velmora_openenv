---
title: Velmora Incident Response OpenEnv
emoji: 🚨
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Velmora Incident Response OpenEnv

Velmora Incident Response OpenEnv is a real-world simulation environment for evaluating AI agents on incident management and operational troubleshooting tasks.

## Tasks

- `easy` — Low-impact UI/frontend incidents
- `medium` — Service degradation, integration failures
- `hard` — High-severity breaches, cascading failures

## Action Space

`investigate` | `fix` | `monitor` | `escalate` | `contain`

## API Endpoints

- `GET /` — service info
- `GET /health` — health check
- `POST /reset` — start a new episode
- `POST /step` — execute an action
- `GET /state` — current environment state
- `GET /tasks` — list tasks and actions
- `GET /grader` — current task score

## Environment Variables

- `MODEL_NAME` — model to use (default: `llama-3.3-70b-versatile`)
- `GROQ_API_KEY` — Groq API key
- `HF_TOKEN` — Hugging Face token
