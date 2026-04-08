import os
import numpy as np
from openai import OpenAI
from velmora_env.environment import IncidentEnv
from velmora_env.models import Action
from velmora_env.grader import grade_task

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

VALID_ACTIONS = os.getenv("VALID_ACTIONS", "investigate,fix,monitor,escalate,contain").split(",")
MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))
BENCHMARK = os.getenv("BENCHMARK", "velmora-incident")


def log_start(task):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step, action_text, reward, done, error=None):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action_text} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)


def log_end(success, steps, score, rewards_list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


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
        return "monitor"

    if severity == "medium":
        if "investigate" not in taken:
            return "investigate"
        if ("gateway" in incident or "vendor" in logs or "payment" in incident or "shipping" in incident or "third" in incident) and "escalate" not in taken:
            return "escalate"
        if "fix" not in taken:
            return "fix"
        return "monitor"

    if severity == "low":
        if ("typo" in incident or "broken" in incident or "link" in incident or "placeholder" in incident or "template" in incident or "search" in incident or "password" in incident):
            if "fix" not in taken:
                return "fix"
            return "monitor"
        if "investigate" not in taken:
            return "investigate"
        if "fix" not in taken:
            return "fix"
        return "monitor"

    return "investigate"


def run_task(task_name):
    env = IncidentEnv(task_name)
    obs = env.reset()

    done = False
    step_count = 0
    rewards_list = np.array([], dtype=np.float32)
    final_score = 0.0
    success = False

    log_start(task_name)

    try:
        while not done and step_count < MAX_STEPS:
            action_text = choose_action(obs)
            action = Action(action=action_text)

            obs, reward, done, info = env.step(action)
            step_count += 1
            rewards_list = np.append(rewards_list, reward.score)

            log_step(step_count, action_text, reward.score, done)

        final_score = grade_task(env, task_name)
        success = final_score > 0.5
    except Exception as e:
        log_step(step_count + 1, "error", 0.0, True, error=str(e))
    finally:
        log_end(success, step_count, final_score, rewards_list.tolist())

    return final_score


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)
