import os
from openai import OpenAI
from velmora_env.environment import IncidentEnv
from velmora_env.models import Action
from velmora_env.grader import grade_task

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("Missing HF_TOKEN")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

VALID_ACTIONS = ["investigate", "fix", "monitor", "escalate", "contain"]
MAX_STEPS = 50
BENCHMARK = "velmora-incident"


def log_start(task):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step, action_text, reward, done, error=None):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action_text} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True
    )


def log_end(success, steps, score, rewards_list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True
    )


def choose_action(obs):
    taken = obs.steps_taken
    severity = obs.severity.lower()
    incident = obs.incident.lower()
    logs = obs.logs.lower()

    if severity == "high":
        if ("breach" in incident or "attack" in incident or "security" in incident or "suspicious" in incident) and "contain" not in taken:
            return "contain"
        if "investigate" not in taken:
            return "investigate"
        if ("vendor" in logs or "external" in logs or "gateway" in incident or "database" in incident or "auth" in incident) and "escalate" not in taken:
            return "escalate"
        if "contain" not in taken:
            return "contain"
        if "fix" not in taken:
            return "fix"
        if "monitor" not in taken:
            return "monitor"
        return "monitor"

    if severity == "medium":
        if "investigate" not in taken:
            return "investigate"
        if ("gateway" in incident or "vendor" in logs or "payment" in incident or "shipping" in incident or "third" in incident) and "escalate" not in taken:
            return "escalate"
        if "fix" not in taken:
            return "fix"
        if "monitor" not in taken:
            return "monitor"
        return "monitor"

    if severity == "low":
        if ("typo" in incident or "broken" in incident or "link" in incident or "placeholder" in incident or "template" in incident or "search" in incident or "password" in incident):
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
    rewards_list = []

    log_start(task_name)

    while not done and step_count < MAX_STEPS:
        action_text = choose_action(obs)
        action = Action(action=action_text)

        obs, reward, done, info = env.step(action)
        step_count += 1
        rewards_list.append(reward.score)

        log_step(step_count, action_text, reward.score, done)

    final_score = grade_task(env, task_name)
    success = final_score > 0.5

    log_end(success, step_count, final_score, rewards_list)
    return final_score


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)