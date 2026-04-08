import os
import json
import time
import sys
import numpy as np
from typing import Any, Dict, List
from openai import OpenAI
from velmora_env.environment import IncidentEnv
from velmora_env.models import Action
from velmora_env.grader import grade_task

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))
BENCHMARK = os.getenv("BENCHMARK", "velmora-incident")
VALID_ACTIONS = ["investigate", "fix", "monitor", "escalate", "contain"]


def emit(tag: str, payload: Dict[str, Any]) -> None:
    parts = " ".join(f"{k}={v}" for k, v in payload.items() if not isinstance(v, (dict, list)))
    print(f"{tag} {parts}", flush=True)


def make_client():
    api_base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")
    client = OpenAI(base_url=api_base_url, api_key=hf_token)
    return client, model_name, api_base_url


def system_prompt() -> str:
    return (
        "You are an expert incident response agent.\n"
        f"Choose exactly one action from: {VALID_ACTIONS}\n\n"
        "Rules:\n"
        "- For LOW severity: go straight to 'fix' for obvious issues (typo, broken link, UI). Only 'investigate' if cause is unclear.\n"
        "- For MEDIUM severity: start with 'investigate', only 'escalate' if logs mention external/vendor/gateway, then 'fix', then 'monitor'.\n"
        "- For HIGH severity: start with 'contain' for security/breach, then 'investigate', then 'escalate', then 'fix', then 'monitor'.\n"
        "- Never repeat an action already taken.\n\n"
        "Respond with exactly one word from the actions list. No explanation."
    )


def format_observation(obs) -> str:
    return (
        f"Incident: {obs.incident}\n"
        f"Severity: {obs.severity}\n"
        f"Hint: {obs.hint}\n"
        f"Logs: {obs.logs}\n"
        f"User Impact: {obs.user_impact}\n"
        f"System Status: {obs.system_status}\n"
        f"Steps already taken: {obs.steps_taken}\n\n"
        f"What is the single best next action?"
    )


def parse_action(raw_text: str) -> str:
    text = raw_text.strip().lower()
    # exact word match first
    for action in VALID_ACTIONS:
        if text == action:
            return action
    # fallback: substring match
    for action in VALID_ACTIONS:
        if action in text:
            return action
    return "investigate"


def format_feedback(reward, obs) -> str:
    return (
        f"Result: reward={reward.score:.2f}, feedback='{reward.feedback}', "
        f"progress={reward.progress:.2f}, stage={obs.current_stage}"
    )


def run_single_task(task: str, client: OpenAI, model_name: str) -> Dict[str, Any]:
    env = IncidentEnv(task_name=task)
    obs = env.reset()

    done = False
    step_idx = 0
    rewards_list = np.array([], dtype=np.float32)
    final_score = 0.0
    success = False

    messages = [{"role": "system", "content": system_prompt()}]

    emit("[START]", {"task": task, "env": BENCHMARK, "model": model_name})

    try:
        while not done and step_idx < MAX_STEPS:
            user_prompt = format_observation(obs)
            messages.append({"role": "user", "content": user_prompt})

            model_error = None
            action_text = "investigate"
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=10,
                )
                raw_content = response.choices[0].message.content or ""
                action_text = parse_action(raw_content)
                messages.append({"role": "assistant", "content": raw_content})
            except Exception as exc:
                model_error = str(exc)

            action = Action(action=action_text)
            obs, reward, done, info = env.step(action)
            step_idx += 1
            rewards_list = np.append(rewards_list, reward.score)

            # feed reward feedback back so model improves next step
            if not done:
                messages.append({"role": "user", "content": format_feedback(reward, obs)})

            emit("[STEP]", {
                "step": step_idx,
                "action": action_text,
                "reward": f"{float(reward.score):.2f}",
                "done": str(done).lower(),
                "error": model_error or "null"
            })

            if len(messages) > 10:
                messages = [messages[0]] + messages[-8:]

        final_score = grade_task(env, task)
        success = final_score > 0.5

    except Exception as e:
        emit("[STEP]", {"step": step_idx + 1, "action": "error",
                        "reward": "0.00", "done": "true", "error": str(e)})

    finally:
        emit("[END]", {
            "success": str(success).lower(),
            "steps": step_idx,
            "score": f"{final_score:.2f}",
            "rewards": ",".join(f"{r:.2f}" for r in rewards_list.tolist())
        })

    return {"task": task, "score": final_score, "steps": step_idx}


def main():
    started_at = time.time()
    try:
        client, model_name, _ = make_client()
    except RuntimeError as e:
        print(f"[ERROR] {e}", flush=True)
        sys.exit(1)

    all_results = []
    for task in TASKS:
        try:
            result = run_single_task(task, client, model_name)
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] task={task} error={e}", flush=True)
            emit("[END]", {"success": "false", "steps": 0,
                           "score": "0.00", "rewards": ""})

    print(f"[SUMMARY] tasks={len(all_results)} runtime={round(time.time() - started_at, 2)}s", flush=True)


if __name__ == "__main__":
    main()
