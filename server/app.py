from fastapi import FastAPI, HTTPException
from velmora_env.environment import IncidentEnv
from velmora_env.models import Action
from velmora_env.grader import grade_task
import uvicorn

app = FastAPI(title="Velmora Incident Response OpenEnv")

env_instance = None


@app.get("/")
def root():
    return {
        "name": "Velmora Incident Response OpenEnv",
        "status": "running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(task: str = "easy"):
    global env_instance
    if task not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail="Invalid task")

    env_instance = IncidentEnv(task_name=task)
    obs = env_instance.reset()

    return {
        "task": task,
        "observation": obs.dict(),
        "state": env_instance.state().dict()
    }


@app.post("/step")
def step(action: Action):
    global env_instance
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    obs, reward, done, info = env_instance.step(action)

    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info,
        "state": env_instance.state().dict()
    }


@app.get("/state")
def state():
    global env_instance
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env_instance.state().dict()


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"id": "easy", "description": "Simple low-impact incidents"},
            {"id": "medium", "description": "Moderately complex incidents"},
            {"id": "hard", "description": "High-severity incidents requiring strategic handling"}
        ],
        "actions": ["investigate", "fix", "monitor", "escalate", "contain"]
    }


@app.get("/grader")
def grader(task: str = "easy"):
    global env_instance
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    score = grade_task(env_instance, task)
    state = env_instance.state()

    total_incidents = len(env_instance.tasks[task]) if task in env_instance.tasks else 0

    return {
        "task": task,
        "score": score,
        "incidents_completed": env_instance.incidents_completed,
        "total_incidents": total_incidents,
        "resources_remaining": state.resources,
        "done": state.done,
        "state": state.dict()
    }


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()