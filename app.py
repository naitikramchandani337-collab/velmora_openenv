from __future__ import annotations

import uuid
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from velmora_env.environment import IncidentEnv
from velmora_env.models import Action
from velmora_env.grader import grade_task

app = FastAPI(title="Velmora Incident Response OpenEnv")

VALID_TASKS = {"easy", "medium", "hard"}
VALID_ACTIONS = {"investigate", "fix", "monitor", "escalate", "contain"}

ENVS: Dict[str, IncidentEnv] = {}


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


class ResetRequest(BaseModel):
    task: str = "easy"

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        if v not in VALID_TASKS:
            raise ValueError(f"task must be one of {sorted(VALID_TASKS)}")
        return v


class StepRequest(BaseModel):
    session_id: str
    action: str

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        if v not in VALID_ACTIONS:
            raise ValueError(f"action must be one of {sorted(VALID_ACTIONS)}")
        return v


@app.get("/")
def root():
    return {
        "name": "Velmora Incident Response OpenEnv",
        "version": "1.0.0",
        "status": "running",
        "tasks": sorted(VALID_TASKS),
        "actions": sorted(VALID_ACTIONS),
        "reward_range": [0.0, 1.0],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(task: str = "easy", body: ResetRequest = None):
    task_name = body.task if body else task
    if task_name not in VALID_TASKS:
        raise HTTPException(status_code=422, detail=f"task must be one of {sorted(VALID_TASKS)}")
    try:
        env = IncidentEnv(task_name=task_name)
        obs = env.reset()
        session_id = str(uuid.uuid4())
        ENVS[session_id] = env
        return {
            "session_id": session_id,
            "task": task_name,
            "observation": obs.dict(),
            "state": env.state().dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(body: StepRequest):
    env = ENVS.get(body.session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="Invalid session_id. Call /reset first.")
    try:
        obs, reward, done, info = env.step(Action(action=body.action))
        return {
            "observation": obs.dict(),
            "reward": reward.dict(),
            "done": done,
            "info": info,
            "state": env.state().dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state(session_id: str):
    env = ENVS.get(session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="Invalid session_id. Call /reset first.")
    try:
        return env.state().dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"id": "easy", "description": "Simple low-impact incidents"},
            {"id": "medium", "description": "Moderately complex incidents"},
            {"id": "hard", "description": "High-severity incidents requiring strategic handling"},
        ],
        "actions": sorted(VALID_ACTIONS),
    }


@app.get("/grader")
def grader(session_id: str, task: str = "easy"):
    if task not in VALID_TASKS:
        raise HTTPException(status_code=422, detail=f"task must be one of {sorted(VALID_TASKS)}")
    env = ENVS.get(session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="Invalid session_id. Call /reset first.")
    try:
        score = grade_task(env, task)
        s = env.state()
        total_incidents = len(env.tasks[task]) if task in env.tasks else 0
        return {
            "task": task,
            "score": score,
            "incidents_completed": env.incidents_completed,
            "total_incidents": total_incidents,
            "resources_remaining": s.resources,
            "done": s.done,
            "state": s.dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
