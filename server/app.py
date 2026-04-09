from __future__ import annotations

import os
import sys
import uuid
from typing import Dict, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from velmora_env.environment import IncidentEnv
from velmora_env.models import Action
from velmora_env.grader import grade_task

app = FastAPI(title="Velmora Incident Response OpenEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_TASKS = {"easy", "medium", "hard"}
VALID_ACTIONS = {"investigate", "fix", "monitor", "escalate", "contain"}

environments: Dict[str, IncidentEnv] = {}


class ResetRequest(BaseModel):
    task: str = "easy"
    seed: Optional[int] = None

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        if v not in VALID_TASKS:
            raise ValueError(f"task must be one of {sorted(VALID_TASKS)}")
        return v


class StepRequest(BaseModel):
    env_id: str
    action: Dict[str, Any]


@app.get("/")
async def root():
    return {
        "name": "Velmora Incident Response OpenEnv",
        "version": "1.0.0",
        "status": "running",
        "tasks": sorted(VALID_TASKS),
        "actions": sorted(VALID_ACTIONS),
        "reward_range": [0.0, 1.0],
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {"id": "easy", "description": "Simple low-impact incidents"},
            {"id": "medium", "description": "Moderately complex incidents"},
            {"id": "hard", "description": "High-severity incidents requiring strategic handling"},
        ],
        "actions": sorted(VALID_ACTIONS),
    }


@app.post("/reset")
async def reset(request: ResetRequest = None):
    try:
        if request is None:
            request = ResetRequest()
        env = IncidentEnv(task_name=request.task)
        obs = env.reset()
        env_id = str(uuid.uuid4())
        environments[env_id] = env
        return {
            "env_id": env_id,
            "task": request.task,
            "observation": obs.dict(),
            "state": env.state().dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: StepRequest):
    try:
        if request.env_id not in environments:
            raise HTTPException(status_code=404, detail="Environment not found. Call /reset first.")
        env = environments[request.env_id]
        action = Action(**request.action)
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.dict(),
            "reward": reward.dict(),
            "done": done,
            "info": info,
            "state": env.state().dict(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state/{env_id}")
async def get_state(env_id: str):
    if env_id not in environments:
        raise HTTPException(status_code=404, detail="Environment not found.")
    return environments[env_id].state().dict()


@app.get("/grader")
async def grader(env_id: str, task: str = "easy"):
    if task not in VALID_TASKS:
        raise HTTPException(status_code=422, detail=f"task must be one of {sorted(VALID_TASKS)}")
    if env_id not in environments:
        raise HTTPException(status_code=404, detail="Environment not found. Call /reset first.")
    try:
        env = environments[env_id]
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


def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
