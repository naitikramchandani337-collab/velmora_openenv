from fastapi import FastAPI
from env.environment import IncidentEnv
from env.models import Action
from baseline.run_baseline import run_task
import uvicorn

app = FastAPI()

env_instance = None

# -------------------------
# RESET
# -------------------------
@app.post("/reset")
def reset(task: str = "easy"):
    global env_instance
    env_instance = IncidentEnv(task_name=task)
    obs = env_instance.reset()
    return obs.dict()


# -------------------------
# STEP
# -------------------------
@app.post("/step")
def step(action: Action):
    global env_instance

    obs, reward, done, info = env_instance.step(action)

    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info
    }


# -------------------------
# STATE
# -------------------------
@app.get("/state")
def state():
    return env_instance.state()


# -------------------------
# TASKS
# -------------------------
@app.get("/tasks")
def tasks():
    return {
        "tasks": ["easy", "medium", "hard"],
        "actions": ["investigate", "fix", "monitor", "escalate", "contain"]
    }


# -------------------------
# BASELINE
# -------------------------
@app.get("/baseline")
def baseline():
    return {
        "easy": run_task("easy"),
        "medium": run_task("medium"),
        "hard": run_task("hard")
    }


# -------------------------
# GRADER
# -------------------------
@app.get("/grader")
def grader():
    return {"message": "Grader handled inside environment step"}


# -------------------------
# REQUIRED MAIN FUNCTION (IMPORTANT)
# -------------------------
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()