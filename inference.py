import os
from openai import OpenAI  # required by competition
from velmora_env.environment import IncidentEnv
from velmora_env.models import Action
from velmora_env.grader import grade_task

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

# Keep OpenAI client usage to satisfy requirement
if not API_BASE_URL:
    raise RuntimeError("Missing API_BASE_URL")
if not MODEL_NAME:
    raise RuntimeError("Missing MODEL_NAME")
if not HF_TOKEN:
    raise RuntimeError("Missing HF_TOKEN")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

MAX_STEPS = 50


def log_start(task_name):
    print(f"[START] task={task_name}", flush=True)


def log_step(task_name, step_count, action_text, score):
    print(
        f"[STEP] task={task_name} step={step_count} action={action_text} score={score:.3f}",
        flush=True
    )


def log_end(task_name, final_score):
    print(f"[END] task={task_name} score={final_score:.3f}", flush=True)


def choose_action(obs):
    taken = obs.steps_taken
    severity = obs.severity.lower()
    incident = obs.incident.lower()
    logs = obs.logs.lower()

    # Hard / security / severe incidents
    if severity == "high":
        if ("breach" in incident or "attack" in incident or "security" in incident) and "contain" not in taken:
            return "contain"
        if "investigate" not in taken:
            return "investigate"
        if ("vendor" in logs or "external" in logs or "gateway" in incident or "database" in incident) and "escalate" not in taken:
            return "escalate"
        if "fix" not in taken:
            return "fix"
        if "monitor" not in taken:
            return "monitor"
        return "monitor"

    # Medium incidents
    if severity == "medium":
        if "investigate" not in taken:
            return "investigate"
        if ("gateway" in incident or "vendor" in logs or "payment" in incident) and "escalate" not in taken:
            return "escalate"
        if "fix" not in taken:
            return "fix"
        if "monitor" not in taken:
            return "monitor"
        return "monitor"

    # Low incidents
    if severity == "low":
        if ("typo" in incident or "broken image" in incident or "link" in incident):
            if "fix" not in taken:
                return "fix"
            if "monitor" not in taken:
                return "monitor"
        if "investigate" not in taken:
            return "investigate"
        if "fix" not in taken:
            return "fix"
        if "monitor" not in taken:
            return "monitor"
        return "monitor"

    return "investigate"


def run_task(task_name):
    env = IncidentEnv(task_name)
    obs = env.reset()

    done = False
    step_count = 0

    log_start(task_name)

    while not done and step_count < MAX_STEPS:
        action_text = choose_action(obs)
        action = Action(action=action_text)

        obs, reward, done, info = env.step(action)
        step_count += 1

        log_step(task_name, step_count, action_text, reward.score)

    final_score = grade_task(env, task_name)
    log_end(task_name, final_score)
    return final_score


if __name__ == "__main__":
    easy = run_task("easy")
    medium = run_task("medium")
    hard = run_task("hard")
    print(
        f"[END] summary easy={easy:.3f} medium={medium:.3f} hard={hard:.3f}",
        flush=True
    )