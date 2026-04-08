from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from velmora_env.environment import IncidentEnv
from velmora_env.models import Action
from velmora_env.grader import grade_task
import uvicorn

app = FastAPI(title="Velmora Incident Response OpenEnv")

env_instance = None

VALID_TASKS = {"easy", "medium", "hard"}
VALID_ACTIONS = {"investigate", "fix", "monitor", "escalate", "contain"}


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
    def validate_task(cls, v):
        if v not in VALID_TASKS:
            raise ValueError(f"task must be one of {sorted(VALID_TASKS)}")
        return v


class GraderRequest(BaseModel):
    task: str = "easy"

    @field_validator("task")
    @classmethod
    def validate_task(cls, v):
        if v not in VALID_TASKS:
            raise ValueError(f"task must be one of {sorted(VALID_TASKS)}")
        return v


class StepRequest(BaseModel):
    action: str

    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        if v not in VALID_ACTIONS:
            raise ValueError(f"action must be one of {sorted(VALID_ACTIONS)}")
        return v


@app.get("/")
def root():
    return {
        "name": "Velmora Incident Response OpenEnv",
        "version": "1.0.0",
        "status": "running",
        "description": "Real-world incident response simulation for evaluating AI agents on operational troubleshooting, escalation, containment, and recovery.",
        "tasks": sorted(VALID_TASKS),
        "actions": sorted(VALID_ACTIONS),
        "reward_range": [0.0, 1.0]
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(body: ResetRequest):
    global env_instance
    try:
        env_instance = IncidentEnv(task_name=body.task)
        obs = env_instance.reset()
        return {"task": body.task, "observation": obs.dict(), "state": env_instance.state().dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(body: StepRequest):
    global env_instance
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    try:
        obs, reward, done, info = env_instance.step(Action(action=body.action))
        return {"observation": obs.dict(), "reward": reward.dict(), "done": done, "info": info, "state": env_instance.state().dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    global env_instance
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    try:
        return env_instance.state().dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"id": "easy", "description": "Simple low-impact incidents"},
            {"id": "medium", "description": "Moderately complex incidents"},
            {"id": "hard", "description": "High-severity incidents requiring strategic handling"}
        ],
        "actions": sorted(VALID_ACTIONS)
    }


@app.get("/grader")
def grader(body: GraderRequest):
    global env_instance
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    try:
        score = grade_task(env_instance, body.task)
        s = env_instance.state()
        total_incidents = len(env_instance.tasks[body.task]) if body.task in env_instance.tasks else 0
        return {
            "task": body.task,
            "score": score,
            "incidents_completed": env_instance.incidents_completed,
            "total_incidents": total_incidents,
            "resources_remaining": s.resources,
            "done": s.done,
            "state": s.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
